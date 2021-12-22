package com.example.testcctv;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class ConnectActivity extends AppCompatActivity {

    private final static String TAG = "ConnectActivity";
    private final Context context = ConnectActivity.this;
    private final static int PERMISSION_ALL = 1;

    private final static String[] PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private EditText ip_text;
    private EditText port_text;
    private ActivityResultLauncher<Intent> resultLauncher;

    private boolean has_permissions = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_connect);

        ip_text = findViewById(R.id.ip_text);
        port_text = findViewById(R.id.port_text);

        getPermissions();

        // Launcher for new activity
        resultLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                new ActivityResultCallback<ActivityResult>() {
                    @Override
                    public void onActivityResult(ActivityResult result) {
                        Log.d(TAG, "Exit CameraActivity.");
                        if (result.getResultCode() == RESULT_CANCELED) {
                            Toast.makeText(context, "Failed to connect to server.", Toast.LENGTH_LONG).show();
                        } else if (result.getResultCode() == RESULT_OK) {
                            Toast.makeText(context, "Finish.", Toast.LENGTH_LONG).show();
                        }
                    }
                });

    }

    // Check permissions
    private boolean checkPermissions() {
        for (String permission: PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(context, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    // Get permissions
    private void getPermissions() {
        has_permissions = checkPermissions();
        if (!has_permissions) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_ALL);
        }
    }

    public void mOnClick(View v) {
        if (v.getId() == R.id.connect_btn) {

            // Get ip address from EditText
            String ip = ip_text.getText().toString();
            if (ip.equals("")) {
                ip = ip_text.getHint().toString();
            }

            // Get port number from EditText
            String port_num = port_text.getText().toString();
            if (port_num.equals("")) {
                port_num = port_text.getHint().toString();
            }
            int port = Integer.parseInt(port_num);

            // Start new activity
            if (has_permissions) {
                if (typeOf(ip).equals("String") && typeOf(port).equals("int")) {
                    Intent intent = new Intent(this, CameraActivity.class);
                    Bundle bundle = new Bundle();
                    bundle.putString("ip", ip);
                    bundle.putInt("port", port);
                    intent.putExtras(bundle);
                    Log.d(TAG, "Start CameraActivity.");
                    resultLauncher.launch(intent);
                } else {
                    Toast.makeText(context, "Wrong address.", Toast.LENGTH_SHORT).show();
                }
            } else {
                Toast.makeText(context, "Permission denied.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    public String typeOf(String arg) {return "String"; }
    public String typeOf(int arg) { return "int"; }
}