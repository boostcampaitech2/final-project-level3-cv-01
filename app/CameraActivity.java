package com.example.testcctv;

import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
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
import java.nio.channels.FileChannel;

import static android.os.Environment.getExternalStorageDirectory;


public class CameraActivity extends AppCompatActivity {

    private final Context context = CameraActivity.this;

    private static final String TAG = "CameraActivity";
    private static final String PATH = getExternalStorageDirectory().getAbsolutePath() + "/DCIM/Camera/";
    private final String saveName = "save.jpg";
    private final String savePath = PATH + saveName;
    private final String sendName = "send.jpg";
    private final String sendPath = PATH + sendName;
    private final int FREQUENCY = 50;
    private final static int BUFFER_SIZE = 1400;

    private Socket socket;

    CameraSurface mSurface;
    Camera mCamera;
    Button mShutter;
    private static CameraActivity instance;
    int cnt;
    int saveCnt=0;

    private boolean isRunning = false;
    private boolean isSending;
    private boolean isRenaming = false;
    private PrintWriter pw;
    private FileInputStream fis;
    private OutputStream os;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        Intent intent = getIntent();
        processCommand(intent);

        instance = this;
        cnt = 0;

        // Check if save directory is exist. If not, make a directory.
        File directory = new File(PATH);
        if (!directory.exists()) {
            directory.mkdirs();
            Log.d(TAG, "Created directory : " + directory.getPath());
        } else {
            Log.d(TAG, "Directory already exist : " + directory.getPath());
        }

        // Instances for camera
        mSurface = findViewById(R.id.preview);
        mCamera = mSurface.mCamera;
        mShutter = findViewById(R.id.shutter_btn);
    }

    // When activity gets started, connect to the server.
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

    // Connect to the server.
    private void connect(InetSocketAddress addr) {
        /*
         * Connect to server.
         * Args:
         *   None
         * Returns:
         *   bool : whether connected or not
         */

        // Connect or float error message
        socket = new Socket();

        new Thread(new Runnable() {
            @Override
            public void run() {
                Log.d(TAG, "Try to connect.");

                // Try to connect to server
                try {
                    socket.connect(addr);
                    Thread.sleep(100);
                } catch (IOException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // If not connected, log error message and quit activity.
                if (!socket.isConnected()) {
                    Log.d(TAG, "Failed to connect.");
                    setResult(RESULT_CANCELED);
                    finish();
                } else {
                    try {
                        os = socket.getOutputStream();
                        pw = new PrintWriter(new BufferedOutputStream(os), true);
                        pw.println("connect");
                        pw.flush();
                        Log.d(TAG, "send connect code");
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

    // Save images and send them to the server.
    private void execute() {
        new Thread(new Runnable() {
            @Override
            public void run() {
                if (isRunning) {
                    isRunning = false;
                    mShutter.setText("START");
                } else {
                    isRunning = true;
                    mShutter.setText("STOP");
                    while (isRunning) {
                        mSurface.mCamera.takePicture(null, null, mPicture);
                        try {
                            Thread.sleep(FREQUENCY);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }).start();
    }

    // Quit activity.
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
                    pw.flush();
                    pw.close();
                    os.close();
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }).start();

        setResult(RESULT_OK);
        finish();
    }

    // Enable shutter
    Camera.AutoFocusCallback mAutoFocus = new Camera.AutoFocusCallback() {
        public void onAutoFocus(boolean success, Camera camera) {
            mShutter.setEnabled(success);
        }
    };

    // Actions when taking a picture
    Camera.PictureCallback mPicture = new Camera.PictureCallback() {

        public void onPictureTaken(byte[] data, Camera camera) {

            new Thread(new Runnable() {
                @Override
                public void run() {

                    saveCnt++;

                    // Save image
                    long beforeTime = System.currentTimeMillis();

                    File file = new File(savePath);
                    Log.d(TAG, "File created.");

                    try {
                        FileOutputStream fos = new FileOutputStream(file);
                        fos.write(data);
                        fos.flush();
                        fos.close();
                        Log.d(TAG, "File was written.");
                    } catch (Exception e) {
                        Log.d(TAG,"Error Occurred : " + e.getMessage());
                        return;
                    }

                    MediaScannerConnection.scanFile(context, new String[] {savePath}, new String[] { "image/jpg" }, null);
                    Log.d(TAG, "image saved : " + savePath);

                    // Rename image
                    File sendFile = new File(sendPath);
                    isRenaming = true;

                    boolean success = false;
                    try {
                        copy(file, sendFile);
                        success = true;
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    if (success) {
                        isRenaming = false;
                        Log.d(TAG, String.format("renamed file {%s}", sendName));
                        MediaScannerConnection.scanFile(context, new String[] {sendPath}, new String[] { "image/jpg" }, null);
                    } else {
                        Log.d(TAG, String.format("Can't rename file {%s}", sendName));
                        return;
                    }

                    long afterTime = System.currentTimeMillis();
                    long secDiffTime = (afterTime - beforeTime);
                    Log.d(TAG,"saveImages - " + "Time spent : " + secDiffTime);
                    Log.d(TAG,"saveImages - " + "number of saved images : " + saveCnt);

                    isSending = true;
                    sendImage();
                    isSending = false;


                }
            }).start();
        }
    };

    // Copy a file.
    public void copy(File src, File dst) throws IOException {
        FileInputStream inStream = new FileInputStream(src);
        FileOutputStream outStream = new FileOutputStream(dst);
        FileChannel inChannel = inStream.getChannel();
        FileChannel outChannel = outStream.getChannel();
        inChannel.transferTo(0, inChannel.size(), outChannel);
        inStream.close();
        outStream.close();
    }

    // Send images to the server.
    private void sendImage() {
        cnt++;

        long beforeTime = System.currentTimeMillis();

        while (isRenaming) {
            continue;
        }

        File file = new File(sendPath);
        long fileSize = file.length();
        final long[] totalReadBytes = {0};
        byte[] buffer = new byte[BUFFER_SIZE];
        int readBytes;

        String code = "image";  // Start of file
        byte[] bcode = code.getBytes();
        try {
            os.write(bcode, 0, bcode.length);
            os.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

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
            String emptyCode = "";
            byte[] bEmptyCode = emptyCode.getBytes();
            os.write(bEmptyCode,0,bEmptyCode.length);
            os.flush();

            String ecode = "EOF";   // End Of File
            byte[] becode = ecode.getBytes();
            os.write(becode, 0, becode.length);
            os.flush();

            fis.close();

            long afterTime = System.currentTimeMillis();
            long secDiffTime = (afterTime - beforeTime);
            Log.d(TAG, "sendImage - " + "Time spent : " + secDiffTime);
            Log.d(TAG, "sendImage - " + "number of sent images : " + cnt);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static CameraActivity getInstance() {
        return instance;
    }
}