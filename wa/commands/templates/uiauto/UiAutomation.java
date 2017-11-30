package ${package_name};

import android.app.Activity;
import android.os.Bundle;
import org.junit.Test;
import org.junit.runner.RunWith;
import android.support.test.runner.AndroidJUnit4;

import android.util.Log;
import android.view.KeyEvent;

// Import the uiautomator libraries
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiScrollable;
import android.support.test.uiautomator.UiSelector;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import com.arm.wa.uiauto.BaseUiAutomation;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    protected Bundle parameters;

    public static String TAG = "${name}";

    @Before
    public void initilize() throws Exception {
        parameters = getParams();
        // Perform any parameter initialization here
    }

    @Test
    public void setup() throws Exception {
        // Optional: Perform any setup required before the main workload
        // is ran, e.g. dismissing welcome screens
    }

    @Test
    public void runWorkload() throws Exception {
	   // The main UI Automation code goes here
    }

    @Test
    public void extractResults() throws Exception {
        // Optional: Extract any relevant results from the workload,
    }

    @Test
    public void teardown() throws Exception {
        // Optional: Perform any clean up for the workload
    }
}
