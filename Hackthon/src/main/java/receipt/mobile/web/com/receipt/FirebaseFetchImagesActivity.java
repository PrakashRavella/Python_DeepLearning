package lab2.mobile.web.com.mobile_lab2;

import android.app.ProgressDialog;
import android.os.Build;
import android.os.Bundle;
import android.view.ViewTreeObserver;
import android.widget.GridLayout;
import android.widget.ImageView;

import com.bumptech.glide.Glide;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

/**
 * Created by gangi on 7/23/2018.
 */

public class FirebaseFetchImagesActivity  extends FireBaseActivity{

    // Database Reference
    DatabaseReference databaseReference;

    // Progress Diolg
    ProgressDialog progressDialog;

    GridLayout gridLayout;
    int height = 1000, width = 1100;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_fetchimages_grid);

        gridLayout = findViewById(R.id.gridLayout);

        databaseReference = FirebaseDatabase.getInstance().getReference("images");
        progressDialog = new ProgressDialog(this);

        progressDialog.setTitle("Kindly wait..");
        progressDialog.setMessage("We are fetching your Images");
        progressDialog.show();

        // Adding Event Listener to Fetch Values
        databaseReference.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                progressDialog.dismiss();

                // Dividing Layout Width, Height by row count, column count
                int layoutWidth = width / 3;
                int layoutHeight = height / 2;

                GridLayout gridLayout = findViewById(R.id.gridLayout);
                // Removing all exisiting Views
                gridLayout.removeAllViews();

                // Iterating the Children Snapshots
                int i = 0;
                for(DataSnapshot snapshot : dataSnapshot.getChildren()){
                    ImageView imageView = new ImageView(getApplicationContext());
                    // Setting Layout Params
                    GridLayout.LayoutParams layoutParams = new GridLayout.LayoutParams();
                    layoutParams.width = layoutWidth;
                    layoutParams.height = layoutHeight;
                    imageView.setLayoutParams(layoutParams);
                    imageView.setId(i);
                    // Gliding an ImageView
                    Glide.with(getApplicationContext()).load(snapshot.getValue(String.class)).into(imageView);
                    // Adding ImageView to Layout
                    gridLayout.addView(imageView, i);
                    i = i+1;
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                progressDialog.dismiss();
            }
        });
    }
}
