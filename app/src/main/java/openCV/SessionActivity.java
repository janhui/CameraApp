package openCV;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

/**
 * Created by josekalladanthyil on 27/05/15.
 */
public class SessionActivity extends Activity {

    public static String SESSION_NAME = "session_name";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_session);
        final EditText session_name = (EditText) findViewById(R.id.textview_session_name);
        Button button_session_submit = (Button) findViewById(R.id.button_session_submit);
        button_session_submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent;
                intent = new Intent(SessionActivity.this, OpenCVActivity.class);
                intent.putExtra(SESSION_NAME, session_name.getText().toString());
                startActivity(intent);
            }
        });

    }
}
