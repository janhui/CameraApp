package Utils;

import android.annotation.SuppressLint;
import android.os.Environment;

import org.opencv.core.Mat;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by josekalladanthyil on 05/06/15.
 */
public class Utils {

    @SuppressLint("SimpleDateFormat")
    public static boolean SaveImage(Mat mat, Mat mIntermediateMat) {
        int iFileOrdinal = 0;
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

}
