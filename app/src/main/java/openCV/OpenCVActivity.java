package openCV;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.LinearLayout.LayoutParams;

import com.firebase.client.DataSnapshot;
import com.firebase.client.Firebase;
import com.firebase.client.FirebaseError;
import com.firebase.client.ValueEventListener;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import Utils.Color;
import Utils.Position;
import Utils.Utils;


public class OpenCVActivity extends Activity implements CvCameraViewListener {

    public static final int VIEW_MODE_RGBA = 0;
    public static final int VIEW_MODE_FILTERED = 1;
    public static int viewMode = VIEW_MODE_RGBA;

    private CascadeClassifier mCascade;

    private boolean bShootNow = false, bDisplayTitle = true;
    private byte[] byteColourTrackCentreHue;
    private double dTextScaleFactor;

    private Position mRedPosition = new Position(0, 0);
    private Position mGreenPosition = new Position(0, 0);
    private Position mBluePosition = new Position(0, 0);
    private boolean redAreaFound = false;
    private boolean greenAreaFound = false;
    private boolean blueAreaFound = false;

    private int iNumberOfCameras = 0;

    private JavaCameraView mOpenCvCameraView0;
    private JavaCameraView mOpenCvCameraView1;

    //FOR FPS AND SAVING IMAGES
    private long lFrameCount = 0, lMilliStart = 0, lMilliNow = 0, lMilliShotTime = 0;

    private Mat mRgba, mIntermediateMat, mMatRed,
            mMatGreen, mMatBlue,
            mHSVMat;

    private Scalar colorRed, colorGreen;
    private Size sSize3;

    private String mSessionName;
    private Firebase myFirebaseRef;
    public static final String mPositionID = "position";


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView0.enableView();

                    if (iNumberOfCameras > 1)
                        mOpenCvCameraView1.enableView();

                    try {
                        // DO FACE CASCADE SETUP

                        Context context = getApplicationContext();
                        InputStream is3 = context.getResources().openRawResource(R.raw.haarcascade_frontalface_default);
                        File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
                        File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_default.xml");

                        FileOutputStream os = new FileOutputStream(cascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        while ((bytesRead = is3.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }

                        is3.close();
                        os.close();

                        mCascade = new CascadeClassifier(cascadeFile.getAbsolutePath());

                        if (mCascade.empty()) {
                            //Log.d(TAG, "Failed to load cascade classifier");
                            mCascade = null;
                        }

                        cascadeFile.delete();
                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        // Log.d(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Firebase.setAndroidContext(this);


        mSessionName = getIntent().getStringExtra(SessionActivity.SESSION_NAME);
        iNumberOfCameras = Camera.getNumberOfCameras();
        myFirebaseRef = new Firebase("https://huddletableapp.firebaseio.com");
        //Log.d(TAG, "called onCreate");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.opencvd2);

        mOpenCvCameraView0 = (JavaCameraView) findViewById(R.id.java_surface_view0);

        if (iNumberOfCameras > 1) {
            mOpenCvCameraView1 = (JavaCameraView) findViewById(R.id.java_surface_view1);
        }

        mRedPosition = new Position(0, 0);
        mGreenPosition = new Position(0, 0);
        mBluePosition = new Position(0, 0);
        redAreaFound = false;
        greenAreaFound = false;
        blueAreaFound = false;

        mOpenCvCameraView0.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView0.setCvCameraViewListener(this);

        mOpenCvCameraView0.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }


    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView0 != null)
            mOpenCvCameraView0.disableView();
        if (iNumberOfCameras > 1)
            if (mOpenCvCameraView1 != null)
                mOpenCvCameraView1.disableView();
    }


    public void onResume() {
        super.onResume();


        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback);
    }


    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView0 != null)
            mOpenCvCameraView0.disableView();
        if (iNumberOfCameras > 1)
            if (mOpenCvCameraView1 != null)
                mOpenCvCameraView1.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.opencvd2, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == R.id.action_rgbpreview) {
            viewMode = VIEW_MODE_RGBA;
            lFrameCount = 0;
            lMilliStart = 0;
        } else if (item.getItemId() == R.id.action_filtered) {
            viewMode = VIEW_MODE_FILTERED;
            lFrameCount = 0;
            lMilliStart = 0;
        }
        return true;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        // TODO Auto-generated method stub
        byteColourTrackCentreHue = new byte[3];
        // green = 60 // mid yellow  27
        byteColourTrackCentreHue[0] = 27;
        byteColourTrackCentreHue[1] = 100;
        byteColourTrackCentreHue[2] = (byte) 255;

        colorRed = new Scalar(255, 0, 0, 255);
        colorGreen = new Scalar(0, 255, 0, 255);

        mHSVMat = new Mat();
        mIntermediateMat = new Mat();
        mMatRed = new Mat();
        mMatGreen = new Mat();
        mMatBlue = new Mat();

        mRgba = new Mat();

        sSize3 = new Size(3, 3);

        DisplayMetrics dm = this.getResources().getDisplayMetrics();
        int densityDpi = dm.densityDpi;
        dTextScaleFactor = ((double) densityDpi / 240.0) * 0.9;

        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);


    }

    @Override
    public void onCameraViewStopped() {
        releaseMats();
    }

    public void releaseMats() {
        mRgba.release();
        mIntermediateMat.release();
        mMatRed.release();
        mMatGreen.release();
        mMatBlue.release();
        mHSVMat.release();

    }


    @Override
    public Mat onCameraFrame(Mat inputFrame) {
        String string = "";
        String sShotText = "";

        // start the timing counter to put the framerate on screen
        // and make sure the start time is up to date, do
        // a reset every 10 seconds
        if (lMilliStart == 0)
            lMilliStart = System.currentTimeMillis();

        if ((lMilliNow - lMilliStart) > 10000) {
            lMilliStart = System.currentTimeMillis();
            lFrameCount = 0;
        }

        inputFrame.copyTo(mRgba);

        switch (viewMode) {

            case VIEW_MODE_RGBA:

                if (bDisplayTitle)
                    ShowTitle("BGR Preview", 1, colorGreen);

                break;
            case VIEW_MODE_FILTERED:


                // start the timing counter to put the framerate on screen
                // and make sure the start time is up to date, do
                // a reset every 10 seconds
                if (lMilliStart == 0) {
                    lMilliStart = System.currentTimeMillis();
                }
                if ((lMilliNow - lMilliStart) > 10000) {
                    lMilliStart = System.currentTimeMillis();
                    lFrameCount = 0;
                }

                inputFrame.copyTo(mRgba);

//            Color Detection
                // Convert the image into an HSV image
                Imgproc.cvtColor(mRgba, mHSVMat, Imgproc.COLOR_RGB2HSV);
                mMatRed = mHSVMat.clone();
                mMatGreen = mHSVMat.clone();
                mMatBlue = mHSVMat.clone();

                Mat lower_red_hue = mRgba.clone();
                Mat upper_red_hue = mRgba.clone();
                Core.inRange(mHSVMat, new Scalar(0, 100, 100), new Scalar(10, 255, 255), lower_red_hue);
                Core.inRange(mHSVMat, new Scalar(160, 100, 100), new Scalar(179, 255, 255), upper_red_hue);
                Core.addWeighted(lower_red_hue, 1.0, upper_red_hue, 1.0, 0.0, mMatRed);
                Core.inRange(mMatGreen, new Scalar(40, 100, 100), new Scalar(75, 255, 255), mMatGreen);
                Core.inRange(mMatBlue, new Scalar(100, 0, 0), new Scalar(120, 255, 255), mMatBlue);

                Imgproc.GaussianBlur(mMatRed, mMatRed, new Size(9, 9), 2, 2);
                Imgproc.GaussianBlur(mMatGreen, mMatGreen, new Size(9, 9), 2, 2);
                Imgproc.GaussianBlur(mMatBlue, mMatBlue, new Size(9, 9), 2, 2);


                lower_red_hue.release();
                upper_red_hue.release();

//        Core.inRange(mMatRed);

                erodeDilate(mMatRed);
                erodeDilate(mMatGreen);
                erodeDilate(mMatBlue);

                Moments RedMoments = Imgproc.moments(mMatRed);
                Moments GreenMoments = Imgproc.moments(mMatGreen);
                Moments BlueMoments = Imgproc.moments(mMatBlue);
                double red_dM01 = RedMoments.get_m01();
                double red_dM10 = RedMoments.get_m10();
                double red_dArea = RedMoments.get_m00();
                double green_dM01 = GreenMoments.get_m01();
                double green_dM10 = GreenMoments.get_m10();
                double green_dArea = GreenMoments.get_m00();
                double blue_dM01 = BlueMoments.get_m01();
                double blue_dM10 = BlueMoments.get_m10();
                double blue_dArea = BlueMoments.get_m00();

                String red;
                String green;
                String blue;
                // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
                if (red_dArea > 10000 && !redAreaFound) {
                    //calculate the position of the ball
                    int posX = (int) (red_dM10 / red_dArea);
                    int posY = (int) (red_dM01 / red_dArea);
                    mRedPosition = new Position(posX, posY);
                    red = "RED AREA";
                    updatePosition(Color.Red, mRedPosition);
//                    redAreaFound = true;
                } else {
                    red = "";
                }
                if (green_dArea > 10000 && !greenAreaFound) {
                    int posX = (int) (green_dM10 / green_dArea);
                    int posY = (int) (green_dM01 / green_dArea);
                    mGreenPosition = new Position(posX, posY);
                    updatePosition(Color.Green, mGreenPosition);
                    green = "Green AREA";
//                    greenAreaFound = true;
                } else {
                    green = "";
                }
                if (blue_dArea > 10000 && !blueAreaFound) {
                    int posX = (int) (blue_dM10 / blue_dArea);
                    int posY = (int) (blue_dM01 / blue_dArea);
                    mBluePosition = new Position(posX, posY);
                    updatePosition(Color.Blue, mBluePosition);
                    blue = "Blue AREA";
//                    blueAreaFound = true;
                } else {
                    blue = "";
                }

                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(mMatRed, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                Imgproc.drawContours(mMatRed, contours, -1, new Scalar(255, 255, 0));
                contours.clear();
                Imgproc.findContours(mMatGreen, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                Imgproc.drawContours(mMatGreen, contours, -1, new Scalar(255, 255, 0));
                contours.clear();
                Imgproc.findContours(mMatBlue, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                Imgproc.drawContours(mMatBlue, contours, -1, new Scalar(255, 255, 0));
                contours.clear();

                Core.addWeighted(mMatBlue, 1.0, mMatGreen, 1.0, 0.0, mMatBlue);
                Core.addWeighted(mMatBlue, 1.0, mMatRed, 1.0, 0.0, mMatRed);
                mRgba = mMatRed;


                showTextAtColorContour(red, mRedPosition);
                showTextAtColorContour(green, mGreenPosition);
                showTextAtColorContour(blue, mBluePosition);

                break;

        }
        // get the time now in every frame
        lMilliNow = System.currentTimeMillis();

        // update the frame counter
        lFrameCount++;

        if (bDisplayTitle) {
            string = String.format("FPS: %2.1f", (float) (lFrameCount * 1000) / (float) (lMilliNow - lMilliStart));

            ShowTitle(string, 2, colorGreen);
        }

        if (bShootNow) {
            // get the time of the attempt to save a screenshot
            lMilliShotTime = System.currentTimeMillis();
            bShootNow = false;

            // try it, and set the screen text accordingly.
            // this text is shown at the end of each frame until
            // 1.5 seconds has elapsed

        }

        if (System.currentTimeMillis() - lMilliShotTime < 1500)
            ShowTitle(sShotText, 3, colorRed);

        return mRgba;
    }

    // show the which area of image has the color ir red area at big red areas.!!
    private void showTextAtColorContour(String color, Position position) {
        Core.putText(mRgba, color, new Point(position.getX(), position.getY()),
                Core.FONT_HERSHEY_SIMPLEX, dTextScaleFactor, colorRed, 2);
    }

    private void updatePosition(final Color color, final Position position) {
        Firebase ref = myFirebaseRef.child(mSessionName).child(color.toString());
        ref.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                Map<String, Position> dataPosition = new HashMap<>();
                dataPosition.put(mPositionID, position);
                myFirebaseRef.child(mSessionName).child(color.toString()).setValue(dataPosition);
            }

            @Override
            public void onCancelled(FirebaseError firebaseError) {

            }
        });
    }

    private void erodeDilate(Mat matrix) {
        //morphological opening (remove small objects from the foreground)
        Imgproc.erode(matrix, matrix, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10)));
        Imgproc.dilate(matrix, matrix, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10)));

        //morphological closing (fill small holes in the foreground)
        Imgproc.dilate(matrix, matrix, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10)));
        Imgproc.erode(matrix, matrix, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(10, 10)));
    }

    public boolean onTouchEvent(final MotionEvent event) {

        bShootNow = true;
        return false; // don't need more than one touch event

    }

    private void ShowTitle(String s, int iLineNum, Scalar color) {
        Core.putText(mRgba, s, new Point(10, (int) (dTextScaleFactor * 60 * iLineNum)),
                Core.FONT_HERSHEY_SIMPLEX, dTextScaleFactor, color, 2);
    }

}