package com.example.testcctv;

import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;
import java.util.List;

// Preview surface class
class CameraSurface extends SurfaceView implements SurfaceHolder.Callback {
    SurfaceHolder mHolder;
    Camera mCamera;
    CameraActivity activity;

    private static final int preH = 1280;   // 1920
    private static final int preW = 720;    // 1080
    private static final int picH = 1280;   // 4032
    private static final int picW = 720;    // 3024

    public CameraSurface(Context context, AttributeSet attrs) {
        super(context, attrs);
        mHolder = getHolder();
        mHolder.addCallback(this);
    }

    // When created, open camera and set preview
    public void surfaceCreated(SurfaceHolder holder) {

        mCamera = Camera.open();

        Camera.Parameters parameters = mCamera.getParameters();
        parameters.setRotation(90);
        parameters.setJpegQuality(50);
        parameters.setPictureSize(picH, picW);
        parameters.setPreviewSize(preH, preW);
        mCamera.setParameters(parameters);

        // Set preview size
        List<Camera.Size> preSizes = parameters.getSupportedPreviewSizes();
        Camera.Size preSize;
        for (int i = 0; i < preSizes.size(); i++) {
            preSize = preSizes.get(i);
            Log.d("MyApplication", "Camera preview size : " + preSize.width + ", " + preSize.height);
        }

        // Set picture size
        List<Camera.Size> picSizes = parameters.getSupportedPictureSizes();
        Camera.Size picSize;
        for (int i = 0; i < picSizes.size(); i++) {
            picSize = picSizes.get(i);
            Log.d("MyApplication", "Camera picture size : " + picSize.width + ", " + picSize.height);
        }

        // Set camera orientation
        activity = CameraActivity.getInstance();
        setCameraDisplayOrientation(activity, Camera.CameraInfo.CAMERA_FACING_BACK, mCamera);
        try {
            mCamera.setPreviewDisplay(mHolder);
        } catch (IOException e) {
            mCamera.release();
            mCamera = null;
        }
    }

    // When surface is destroyed, destroy camera object too
    public void surfaceDestroyed(SurfaceHolder holder) {
        if (mCamera != null) {
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
        }
    }

    // Logic to determine the size of preview
    public void surfaceChanged(SurfaceHolder holder, int format, int width,
                               int height) {
        Camera.Parameters params = mCamera.getParameters();
        List<Camera.Size> arSize = params.getSupportedPreviewSizes();
        if (arSize == null) {
            params.setPreviewSize(width, height);
        } else {
            int diff = 10000;
            Camera.Size opti = null;
            for (Camera.Size s : arSize) {
                if (Math.abs(s.height - height) < diff) {
                    diff = Math.abs(s.height - height);
                    opti = s;

                }
            }
            params.setPreviewSize(opti.width, opti.height);
        }
        mCamera.setParameters(params);
        mCamera.startPreview();
    }

    // Logic to determine the orientation of the camera
    public static void setCameraDisplayOrientation(Activity activity,
                                                   int cameraId, android.hardware.Camera camera) {
        android.hardware.Camera.CameraInfo info =
                new android.hardware.Camera.CameraInfo();
        android.hardware.Camera.getCameraInfo(cameraId, info);
        int rotation = activity.getWindowManager().getDefaultDisplay()
                .getRotation();
        int degrees = 0;
        switch (rotation) {
            case Surface.ROTATION_0: degrees = 0; break;
            case Surface.ROTATION_90: degrees = 90; break;
            case Surface.ROTATION_180: degrees = 180; break;
            case Surface.ROTATION_270: degrees = 270; break;
        }

        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (info.orientation + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }
        camera.setDisplayOrientation(result);
    }
}
