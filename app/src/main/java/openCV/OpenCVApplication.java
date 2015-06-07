package openCV;

import android.app.Application;

import com.firebase.client.Firebase;

/**
 * Created by josekalladanthyil on 27/05/15.
 */
public class OpenCVApplication extends Application {
    private Firebase myFirebaseRef;
    @Override
    public void onCreate() {
        super.onCreate();
        Firebase.setAndroidContext(this);
        myFirebaseRef = new Firebase("https://huddletableapp.firebaseio.com");
    }

    public Firebase getMyFirebaseRef() {
        return myFirebaseRef;
    }
}
