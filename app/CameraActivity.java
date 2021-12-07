package com.example.testcctv;

import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Set;

import static android.os.Environment.getExternalStorageDirectory;


public class CameraActivity extends AppCompatActivity {

    private final Context context = CameraActivity.this;

    private static final String TAG = "CameraActivity";
    private static final String PATH = getExternalStorageDirectory().getAbsolutePath() + "/DCIM/Camera/";

    private Socket socket;

    CameraSurface mSurface;
    Camera mCamera;
//    MediaRecorder mRecorder;
//    SurfaceHolder mSurfaceHolder;
    Button mShutter;
//    Button mRecord;
    private static CameraActivity instance;
    int cnt;

    private boolean isRunning = false;
    private PrintWriter pw;
    private SendThread sender;
    private FileInputStream fis;
    private OutputStream os;


    boolean isRecording = false;
    private boolean is_connected = false;
//    boolean isPlaying = false;
//    boolean hasVideo = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        Intent intent = getIntent();
        processCommand(intent);



        instance = this;
        cnt = 0;

        // check if save directory is exist
        File directory = new File(PATH);
        if (!directory.exists()) {
            directory.mkdirs();
            Log.d(TAG, "Created directory : " + directory.getPath());
        } else {
            Log.d(TAG, "Directory already exist : " + directory.getPath());
        }

        mSurface = findViewById(R.id.preview);
        mCamera = mSurface.mCamera;
        mShutter = findViewById(R.id.shutter_btn);

//        // handle auto focus
//        findViewById(R.id.focus).setOnClickListener(
//                v -> {
//                    mShutter.setEnabled(false);
//                    mSurface.mCamera.autoFocus(mAutoFocus);
//                });
//
//        // handle shutter button
//        mShutter = findViewById(R.id.shutter);
//        mShutter.setOnClickListener(v -> startRecording());
    }

    private void processCommand(Intent intent) {
        Bundle bundle = intent.getExtras();
        String ip = bundle.getString("ip");
        int port = bundle.getInt("port");

        Log.d(TAG, "Start CameraActivity.");

        InetSocketAddress addr = new InetSocketAddress(ip, port);
        connect(addr);

        Log.d(TAG, "Connected to server.");
        Log.d(TAG, "Complete process.");
    }

    private void connect(InetSocketAddress addr) {
        /*
         * Connect to server.
         * Args:
         *   None
         * Returns:
         *   bool : whether connected or not
         */

        // connect or float error message
        socket = new Socket();

        new Thread(new Runnable() {
            @Override
            public void run() {
                Log.d(TAG, "Try to connect.");
                // try to connect to server
                try {
                    socket.connect(addr);
                    Thread.sleep(100);
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                if (!socket.isConnected()) {
                    Log.d(TAG, "Failed to connect.");
                    setResult(RESULT_CANCELED);
                    finish();
                } else {
                    try {
                        Log.d(TAG, "Try to create os, pw.");
                        os = socket.getOutputStream();
                        pw = new PrintWriter(new BufferedOutputStream(os), true);
                        Log.d(TAG, "os, pw created.");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
        }).start();
    }

    public void mOnClick(View v) {
        switch (v.getId()) {
            case R.id.focus_btn:
                mShutter.setEnabled(false);
                mSurface.mCamera.autoFocus(mAutoFocus);
                break;
            case R.id.shutter_btn:
                execute();
                break;
            case R.id.quit_btn:
                quit();
                break;
        }
    }

    private void execute() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                if (isRecording) {
                    isRecording = false;
                    mShutter.setText("START");
                } else {
                    long beforeTime = System.currentTimeMillis();
                    isRecording = true;
                    mShutter.setText("STOP");
                    while (isRecording) {
                        cnt++;
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        mSurface.mCamera.takePicture(null, null, mPicture);
                        Log.d(TAG, "cnt : " + cnt);
//                        sendImage();
                    }
                    long afterTime = System.currentTimeMillis();
                    long secDiffTime = (afterTime - beforeTime);
                    Log.d(TAG, "Time spent : " + secDiffTime);
                }
            }
        }).start();
    }

    public void quit() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                if (isRunning) {
                    isRunning = false;
                }

                try {
                    String code = "close";
                    pw.println(code);
                    pw.close();
                    os.close();
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }

    public void sendImage() {

        sender = new SendThread("Sender" + cnt);
        sender.setDaemon(true);
        sender.start();
        isRunning = true;
        Log.d(TAG, "File transfer completed.");
    }

    class SendThread extends Thread {

        public String name;

        public SendThread(String name) {
            this.name = name;
        }

        @Override
        public void run() {
            String fileName = String.format("save.jpg", cnt);
            String path = PATH + fileName;
            File file = new File(path);

            if (file.exists()) {
                long fileSize = file.length();
                final long[] totalReadBytes = {0};
                byte[] buffer = new byte[32768];
                int readBytes;
//            final double[] startTime = {0};

                String code = "image";
                //                    pw = new PrintWriter(new BufferedOutputStream(socket.getOutputStream()), true);
                pw.println(code);
                pw.flush();
                Log.d(TAG, "Send code : " + code);

                try {
                    fis = new FileInputStream(file);
                    if (!socket.isConnected()) {
                        Log.d(TAG, "socket is not connected.");
                    }

//                startTime[0] = System.currentTimeMillis();
                    os = socket.getOutputStream();
                    while ((readBytes = fis.read(buffer)) > 0) {
//                        set.add(readBytes);
                        os.write(buffer, 0, readBytes);
                        totalReadBytes[0] += readBytes;
                        Log.d(TAG, "In progress: " + totalReadBytes[0] + "/"
                                + fileSize + " Byte(s) ("
                                + (totalReadBytes[0] * 100 / fileSize) + " %)");
                    }
//                    Log.d(TAG, "readBytes set : " + set.toString());
                    pw.println("");
                    pw.flush();

                    fis.close();
//                os.close();

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

//            code = "end";
//            pw.println(code);
//            pw.flush();
//            Log.d(TAG, "Send code : " + code);
            } else {
                Log.d(TAG, String.format("file not exits {%s}", fileName));
            }
        }
    }

//    private boolean checkCameraHardware(Context context) {
//        if (context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)){
//            // this device has a camera
//            return true;
//        } else {
//            // no camera on this device
//            return false;
//        }
//    }

    // enable shutter
    Camera.AutoFocusCallback mAutoFocus = new Camera.AutoFocusCallback() {
        public void onAutoFocus(boolean success, Camera camera) {
            mShutter.setEnabled(success);
        }
    };

    // save image file
    Camera.PictureCallback mPicture = new Camera.PictureCallback() {

        public void onPictureTaken(byte[] data, Camera camera) {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    Thread t = Thread.currentThread();
                    Log.d(TAG + t, "Thread " + t.getId() + " started.");

                    Set<Thread> threadSet = Thread.getAllStackTraces().keySet();
                    Log.d(TAG + t, "Numerber of Threads : " + threadSet.size());

                    cnt += 1;
                    String fileName = String.format("save.jpg", cnt);
                    String path = PATH + fileName;
                    File file = new File(path);

//                    if (file.exists()) {
//                        String send_fileName = String.format("send.jpg", cnt);
//                        String send_path = PATH + send_fileName;
//                        File send_file = new File(send_path);
//
//                        boolean success = file.renameTo(send_file);
//                        if (!success) {
//                            Log.d(TAG + t, String.format("Can't rename file {%s}", fileName));
//                        }
//                    }

                    Log.d(TAG + t, "File created.");
                    try {
                        FileOutputStream fos = new FileOutputStream(file);
                        fos.write(data);
                        fos.flush();
                        fos.close();
                        Log.d(TAG + t, "File was written.");
                    } catch (Exception e) {
                        Log.d(TAG,"Error Occurred : " + e.getMessage());
                        return;
                    }

//                    MediaScannerConnection.scanFile(context, new String[] { file.getPath() }, new String[] { "image/jpg" }, null);
                    Log.d(TAG, "image saved : " + path);

//                    Log.d(TAG, "Thread " + t.getId() + " finished.");

                    //////////////////////////////

                    long fileSize = file.length();
                    final long[] totalReadBytes = {0};
                    byte[] buffer = new byte[32768];
                    int readBytes;

                    String code = "image";
                    pw.println(code);
                    pw.flush();
                    Log.d(TAG, "Send code : " + code);

                    try {
                        fis = new FileInputStream(file);
                        if (!socket.isConnected()) {
                            Log.d(TAG, "socket is not connected.");
                        }

                        os = socket.getOutputStream();
                        while ((readBytes = fis.read(buffer)) > 0) {
                            os.write(buffer, 0, readBytes);
                            totalReadBytes[0] += readBytes;
                            Log.d(TAG, "In progress: " + totalReadBytes[0] + "/"
                                    + fileSize + " Byte(s) ("
                                    + (totalReadBytes[0] * 100 / fileSize) + " %)");
                        }
                        pw.println("");
                        pw.flush();

                        fis.close();

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
            }).start();
        }
    };

    public static CameraActivity getInstance() {
        return instance;
    }
}