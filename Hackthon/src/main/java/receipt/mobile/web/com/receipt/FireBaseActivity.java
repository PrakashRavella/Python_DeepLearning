package lab2.mobile.web.com.mobile_lab2;

import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.OnProgressListener;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.UUID;

/**
 * Created by gangi on 7/23/2018.
 */


/**
 * This Activity is used to store images in Firebase
 */

public class FireBaseActivity extends AppCompatActivity{

    private Button btnChoose, btnUpload, btnClickPic;
    private ImageView showImageView;

    // FilePath to get the Image on click of Choose
    private Uri filePath;

    // Pick Image Request Code.
    private final int PICK_IMAGE_REQUEST = 71;

    // Image Capture Request Code
    private final int REQUEST_IMAGE_CAPTURE = 1;

    // Firebase Storage
    FirebaseStorage firebaseStorage;
    StorageReference storageReference;

    // Progress Diolg
    ProgressDialog progressDialog;

    // Database Reference
    DatabaseReference databaseReference;

    // Byte Data
    byte[] byteData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_firebase);

        //Initializing Views
        btnChoose = findViewById(R.id.btnChoose);
        btnUpload = findViewById(R.id.btnUpload);
        btnClickPic = findViewById(R.id.btnClick);
        showImageView = findViewById(R.id.showImage);

        // Initializing FirebaseStorage, StorageReference's
        firebaseStorage = FirebaseStorage.getInstance();
        storageReference = firebaseStorage.getReference();

        // Initializing progressDialog
        progressDialog = new ProgressDialog(this);

        // Initializing databaseReference
        databaseReference = FirebaseDatabase.getInstance().getReference("images");


        // On Click Listeners
        // 1) On Click of Choose.
        btnChoose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                    // Calling a Private Method.
                    selectImage();
            }
        });


        // 2) On Click of Click Pic
        btnClickPic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                    clickPicture();
            }
        });


        // 3) On Click of Upload
        btnUpload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                    uploadImage();
            }
        });
    }


    // Method to choose the Picture from Gallery
    private void selectImage() {
        /**
         * When this method is called, a new Intent instance is created. The intent type is set to image, and its action is
         * set to get some content. The intent creates an image chooser dialog that allows the user to browse through the device
         * gallery to select the image. startActivityForResult is used to receive the result, which is the selected image.
         * To display this image, we'll make use of a method called onActivityResult.
         */
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_REQUEST);
    }

    // On Activity Result for Displaying Image
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // If Request Code is Pick Image & Data is present - Get the Data and Convert to Bitmap
        if(requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK
                && data != null && data.getData() != null ) {
            filePath = data.getData();
            try {
                // Getting from Internal Storage using MediaStore.Images.Media
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), filePath);
                // Setting the Image BitMap to Image View
                showImageView.setImageBitmap(bitmap);
                // Enabling Upload Button to upload to Firebase
                btnUpload.setVisibility(View.VISIBLE);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        // On Click of Image Capture
        /**
         * The Android Camera application encodes the photo in the return Intent delivered to onActivityResult()
         * as a small Bitmap in the extras, under the key "data".
         * The following code retrieves this image and displays it in an ImageView.
         */
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            // Converting BitMap into Bytes to save into Firebase
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            // Compressing to JPEG
            imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos);
            byteData = baos.toByteArray();

            // Setting the Image BitMap to Image View
            showImageView.setImageBitmap(imageBitmap);
            // Enabling Upload Button to upload to Firebase
            btnUpload.setVisibility(View.VISIBLE);
        }
    }


    // Method to Upload Image to Firebase
    private void uploadImage(){
        // If File Path is Not Null, then Proceed
        if(filePath != null){
            // Showing a Progress Diolg
            progressDialog.setTitle("Uploading your Image, Please wait");
            progressDialog.show();

            // Creating a Random UID and Storing to Firebase Path under /images section
            StorageReference storageRef = storageReference.child("images/"+ UUID.randomUUID().toString());

            storageRef.putFile(filePath).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                    // Dismissing Progress Diolg
                    progressDialog.dismiss();
                    // On Success Listener, making a Toast once Successful
                    Toast.makeText(FireBaseActivity.this, "Upload Successful !!!", Toast.LENGTH_SHORT).show();

                    //adding an upload to firebase database
                    String uploadId = databaseReference.push().getKey();
                    databaseReference.child(uploadId).setValue(taskSnapshot.getDownloadUrl().toString());
                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    // Dismissing Progress Diolg
                    progressDialog.dismiss();
                    // On Failure Listener, making a Toast with Error
                    Toast.makeText(FireBaseActivity.this, "Upload Failed - "+e.getMessage(),
                            Toast.LENGTH_SHORT).show();
                }
            }).addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onProgress(UploadTask.TaskSnapshot taskSnapshot) {
                    // Reference: https://stackoverflow.com/questions/38278249/update-progress-bar-with-firebase-uploading
                    double progress = (100.0*taskSnapshot.getBytesTransferred()/taskSnapshot
                            .getTotalByteCount());
                    // Showing the Percentage of Uploading done
                    progressDialog.setMessage("Uploaded "+(int)progress+"%");
                }
            });
        }else if(byteData != null){
            // Showing a Progress Diolg
            progressDialog.setTitle("Uploading your Image, Please wait");
            progressDialog.show();

            // Creating a Random UID and Storing to Firebase Path under /images section
            StorageReference storageRef = storageReference.child("images/"+ UUID.randomUUID().toString());

            storageRef.putBytes(byteData).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
                    // Dismissing Progress Diolg
                    progressDialog.dismiss();
                    // On Success Listener, making a Toast once Successful
                    Toast.makeText(FireBaseActivity.this, "Upload Successful !!!", Toast.LENGTH_SHORT).show();

                    String uploadId = databaseReference.push().getKey();
                    databaseReference.child(uploadId).setValue(taskSnapshot.getDownloadUrl().toString());
                }
            }).addOnFailureListener(new OnFailureListener() {
                @Override
                public void onFailure(@NonNull Exception e) {
                    // Dismissing Progress Diolg
                    progressDialog.dismiss();
                    // On Failure Listener, making a Toast with Error
                    Toast.makeText(FireBaseActivity.this, "Upload Failed - "+e.getMessage(),
                            Toast.LENGTH_SHORT).show();
                }
            }).addOnProgressListener(new OnProgressListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onProgress(UploadTask.TaskSnapshot taskSnapshot) {
                    // Reference: https://stackoverflow.com/questions/38278249/update-progress-bar-with-firebase-uploading
                    double progress = (100.0*taskSnapshot.getBytesTransferred()/taskSnapshot
                            .getTotalByteCount());
                    // Showing the Percentage of Uploading done
                    progressDialog.setMessage("Uploaded "+(int)progress+"%");
                }
            });
        } else{
            Toast.makeText(FireBaseActivity.this, "Nothing to Upload !!",
                    Toast.LENGTH_SHORT).show();
        }
    }

    // On Click of Click Pic
    private void clickPicture(){
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
    }

}
