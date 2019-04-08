package lab2.mobile.web.com.mobile_lab2;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;


public class MainActivity extends AppCompatActivity {

    Button firebaseBtn, fetchFirebaseBtn, instaPicsBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        firebaseBtn = findViewById(R.id.firebase);
        fetchFirebaseBtn = findViewById(R.id.fetchFirebase);
        instaPicsBtn = findViewById(R.id.instaPics);

        // On Click Listener
        firebaseBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Redirecting to Firebase Intent
                Intent intent = new Intent(MainActivity.this, FireBaseActivity.class);
                startActivity(intent);
            }
        });

        // On Click Listener
        fetchFirebaseBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Redirecting to Fetch Firebase Images Intent
                Intent intent = new Intent(MainActivity.this, FirebaseFetchImagesActivity.class);
                startActivity(intent);
            }
        });

        // On Click Listener
        instaPicsBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Redirecting to Instagram Intent
                Intent intent = new Intent(MainActivity.this, InstagramFetchProfilePicActivity.class);
                startActivity(intent);
            }
        });
    }
}
