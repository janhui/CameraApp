package openCV;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.DisplayMetrics;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.LinearLayout.LayoutParams;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class OpenCVActivity extends Activity implements CvCameraViewListener {

    public static final int VIEW_MODE_COLOR_DETECTION = 9;
    public static int viewMode = VIEW_MODE_COLOR_DETECTION;

    private CascadeClassifier mCascade;

    private boolean bShootNow = false, bDisplayTitle = true;
    private byte[] byteColourTrackCentreHue;
    private double dTextScaleFactor;

    private int iLastX = 0;
    private int iLastY = 0;

    private Point pt, pt1, pt2;

    private int x, y, radius, iMinRadius, iMaxRadius, iCannyLowerThreshold,
            iCannyUpperThreshold, iAccumulator, iMaxFaceHeight, iMaxFaceHeightIndex,
            iFileOrdinal = 0, iNumberOfCameras = 0;

    private JavaCameraView mOpenCvCameraView0;
    private JavaCameraView mOpenCvCameraView1;

    private List<Byte> byteStatus;
    private List<Integer> iHueMap, channels;
    private List<Float> ranges;
    private List<Point> pts, corners, cornersThis, cornersPrev;
    private List<MatOfPoint> contours;

    private long lFrameCount = 0, lMilliStart = 0, lMilliNow = 0, lMilliShotTime = 0;

    private Mat mRgba, mGray, mIntermediateMat, mMatRed, mMatGreen, mMatBlue, mROIMat,
            mMatRedInv, mMatGreenInv, mMatBlueInv, mHSVMat, mErodeKernel, mContours,
            lines, mFaceDest, mFaceResized, matOpFlowPrev, matOpFlowThis,
            matFaceHistogramPrevious, matFaceHistogramThis, mHist;

    private MatOfFloat mMOFerr, MOFrange;
    private MatOfRect faces;
    private MatOfByte mMOBStatus;
    private MatOfPoint2f mMOP2f1, mMOP2f2, mMOP2fptsPrev, mMOP2fptsThis, mMOP2fptsSafe;
    private MatOfPoint2f mApproxContour;
    private MatOfPoint MOPcorners;
    private MatOfInt MOIone, histSize;

    private Rect rect, rDest;

    private Scalar colorRed, colorGreen;
    private Size sSize, sSize3, sSize5, sMatSize;
    private String string, sShotText;


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

        iNumberOfCameras = Camera.getNumberOfCameras();

        //Log.d(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.opencvd2);

        mOpenCvCameraView0 = (JavaCameraView) findViewById(R.id.java_surface_view0);

        if (iNumberOfCameras > 1)
            mOpenCvCameraView1 = (JavaCameraView) findViewById(R.id.java_surface_view1);

        mOpenCvCameraView0.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView0.setCvCameraViewListener(this);

        mOpenCvCameraView0.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

        if (iNumberOfCameras > 1) {
            mOpenCvCameraView1.setVisibility(SurfaceView.GONE);
            mOpenCvCameraView1.setCvCameraViewListener(this);
            mOpenCvCameraView1.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
        }

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

        viewMode = VIEW_MODE_COLOR_DETECTION;

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
        if (item.getItemId() == R.id.action_info) {
            Intent myIntent1 = new Intent(Intent.ACTION_VIEW, Uri.parse("http://www.barrythomas.co.uk/machinevision.html"));
            startActivity(myIntent1);
        } else if (item.getItemId() == R.id.action_color_detection) {
            viewMode = VIEW_MODE_COLOR_DETECTION;
        } else
            Toast.makeText(getApplicationContext(), "Sadly, your device does not have a second camera",
                    Toast.LENGTH_LONG).show();
//        }

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
        byteStatus = new ArrayList<Byte>();

        channels = new ArrayList<Integer>();
        channels.add(0);
        colorRed = new Scalar(255, 0, 0, 255);
        colorGreen = new Scalar(0, 255, 0, 255);
        contours = new ArrayList<MatOfPoint>();
        corners = new ArrayList<Point>();
        cornersThis = new ArrayList<Point>();
        cornersPrev = new ArrayList<Point>();

        faces = new MatOfRect();

        histSize = new MatOfInt(25);

        iHueMap = new ArrayList<Integer>();
        iHueMap.add(0);
        iHueMap.add(0);
        lines = new Mat();

        mApproxContour = new MatOfPoint2f();
        mContours = new Mat();
        mHist = new Mat();
        mGray = new Mat();
        mHSVMat = new Mat();
        mIntermediateMat = new Mat();
        mMatRed = new Mat();
        mMatGreen = new Mat();
        mMatBlue = new Mat();
        mMatRedInv = new Mat();
        mMatGreenInv = new Mat();
        mMatBlueInv = new Mat();
        MOIone = new MatOfInt(0);

        MOFrange = new MatOfFloat(0f, 256f);
        mMOP2f1 = new MatOfPoint2f();
        mMOP2f2 = new MatOfPoint2f();
        mMOP2fptsPrev = new MatOfPoint2f();
        mMOP2fptsThis = new MatOfPoint2f();
        mMOP2fptsSafe = new MatOfPoint2f();
        mMOFerr = new MatOfFloat();
        mMOBStatus = new MatOfByte();
        MOPcorners = new MatOfPoint();
        mRgba = new Mat();
        mROIMat = new Mat();
        mFaceDest = new Mat();
        mFaceResized = new Mat();
        matFaceHistogramPrevious = new Mat();
        matFaceHistogramThis = new Mat();
        matOpFlowThis = new Mat();
        matOpFlowPrev = new Mat();

        pt = new Point(0, 0);
        pt1 = new Point(0, 0);
        pt2 = new Point(0, 0);

        pts = new ArrayList<>();

        ranges = new ArrayList<>();
        ranges.add(50.0f);
        ranges.add(256.0f);
        rect = new Rect();
        rDest = new Rect();

        sMatSize = new Size();
        sSize = new Size();
        sSize3 = new Size(3, 3);
        sSize5 = new Size(5, 5);

        string = "";

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
        mGray.release();
        mMatRed.release();
        mMatGreen.release();
        mMatBlue.release();
        mROIMat.release();
        mMatRedInv.release();
        mMatGreenInv.release();
        mMatBlueInv.release();
        mHSVMat.release();
        mErodeKernel.release();
        mContours.release();
        lines.release();
        faces.release();
        MOPcorners.release();
        mMOP2f1.release();
        mMOP2f2.release();
        mApproxContour.release();

    }

    @Override
    public Mat onCameraFrame(Mat inputFrame) {
        iMinRadius = 20;
        iMaxRadius = 400;
        iCannyLowerThreshold = 50;
        iCannyUpperThreshold = 180;
        iAccumulator = 300;
        mErodeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, sSize3);

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
        sMatSize.width = mRgba.width();
        sMatSize.height = mRgba.height();

        switch (viewMode) {


//            Color Detection
//todo:
            case VIEW_MODE_COLOR_DETECTION :
                // Convert the image into an HSV image
                Imgproc.cvtColor(mRgba, mHSVMat, Imgproc.COLOR_RGB2HSV);
                Mat mHsvt = mRgba.clone();

                Mat lower_red_hue = mRgba.clone();
                Mat upper_red_hue = mRgba.clone();
                Core.inRange(mHSVMat, new Scalar(0,100,100), new Scalar(10,255,255), lower_red_hue);
                Core.inRange(mHSVMat, new Scalar(160,100,100), new Scalar(179,255,255), upper_red_hue);
                Core.addWeighted(lower_red_hue,1.0,upper_red_hue,1.0,0.0,mHsvt);
                Imgproc.GaussianBlur(mHsvt, mHsvt, new Size(9,9), 2, 2);


                //morphological opening (remove small objects from the foreground)
                Imgproc.erode(mHsvt, mHsvt, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));
                Imgproc.dilate(mHsvt, mHsvt, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));

                //morphological closing (fill small holes in the foreground)
                Imgproc.dilate(mHsvt, mHsvt, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));
                Imgproc.erode(mHsvt, mHsvt, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5)));
                Moments oMoments = Imgproc.moments(mHsvt);
                double dM01 = oMoments.get_m01();
                double dM10 = oMoments.get_m10();
                double dArea = oMoments.get_m00();

                String red;
                // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
                if (dArea > 10000)
                {
                    //calculate the position of the ball
                    int posX = (int)(dM10 / dArea);
                    int posY = (int)(dM01 / dArea);


                    iLastX = posX;
                    iLastY = posY;
                    red = "RED AREA";
                } else {
                    red = "";
                }

                List<MatOfPoint> contours = new ArrayList<>();
                Imgproc.findContours(mHsvt, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                Imgproc.drawContours(mHsvt, contours, -1, new Scalar(255,255,0));


                mRgba = mHsvt;
                //Draw a red line from the previous point to the current point
                Core.putText(mRgba, red, new Point(iLastX, iLastY),
                        Core.FONT_HERSHEY_SIMPLEX, dTextScaleFactor, colorRed, 2);

                if (bDisplayTitle)
                    ShowTitle("Color Detection", 1, colorGreen);
                break;
        }

        // get the time now in every frame
        lMilliNow = System.currentTimeMillis();

        // update the frame counter
        lFrameCount++;

        if (bDisplayTitle) {
            string = String.format("FPS: %2.1f", (float) (lFrameCount * 1000) / (float) (lMilliNow - lMilliStart));

            ShowTitle(string, 2, colorRed);
        }

        if (bShootNow) {
            // get the time of the attempt to save a screenshot
            lMilliShotTime = System.currentTimeMillis();
            bShootNow = false;

            // try it, and set the screen text accordingly.
            // this text is shown at the end of each frame until 
            // 1.5 seconds has elapsed
            if (SaveImage(mRgba)) {
                sShotText = "SCREENSHOT SAVED";
            } else {
                sShotText = "SCREENSHOT FAILED";
            }

        }

        if (System.currentTimeMillis() - lMilliShotTime < 1500)
            ShowTitle(sShotText, 3, colorRed);

        return mRgba;
    }

    public boolean onTouchEvent(final MotionEvent event) {

        bShootNow = true;
        return false; // don't need more than one touch event

    }

    @SuppressLint("SimpleDateFormat")
    public boolean SaveImage(Mat mat) {

        Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_RGBA2BGR, 3);

        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

        String filename = "OpenCV_";
        SimpleDateFormat fmt = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
        Date date = new Date(System.currentTimeMillis());
        String dateString = fmt.format(date);
        filename += dateString + "-" + iFileOrdinal;
        filename += ".png";

        File file = new File(path, filename);

        Boolean bool;
        filename = file.toString();
        bool = Highgui.imwrite(filename, mIntermediateMat);

        //if (bool == false)
        //Log.d("Baz", "Fail writing image to external storage");

        return bool;

    }


    private void ShowTitle(String s, int iLineNum, Scalar color) {
        Core.putText(mRgba, s, new Point(10, (int) (dTextScaleFactor * 60 * iLineNum)),
                Core.FONT_HERSHEY_SIMPLEX, dTextScaleFactor, color, 2);
    }

}