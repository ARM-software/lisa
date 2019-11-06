/*    Copyright 2018 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

package com.arm.wa.uiauto.googlemaps;

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

import static com.arm.wa.uiauto.BaseUiAutomation.Direction.DOWN;
import static com.arm.wa.uiauto.BaseUiAutomation.Direction.UP;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    protected Bundle parameters;
    protected String packageID;
    protected boolean offlineMode;

    public static String TAG = "googlemaps";

    @Before
    public void initialize() throws Exception {
        parameters = getParams();
        packageID = getPackageID(parameters);
        offlineMode = parameters.getBoolean("offline_mode");
    }

    @Test
    public void setup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        runApplicationSetup();
    }

    @Test
    public void runWorkload() throws Exception {
        // Search for and select Cambridge train station
        search("Cambridge Station CBG, England", "search_omnibox_text_box");
        selectSearchResultContaining("[CBG]");
        dismissLocationTutorial();
        sleep(3);

        /** On newer version the location info at bottom of screen interferes with swiping tests so remove by 
        swiping down. Check if new version by seeing if view switcher is present **/
        UiObject viewSwitcher =
            mDevice.findObject(new UiSelector().className("android.widget.ViewSwitcher"));
        UiObject cambridgeTextView =
            mDevice.findObject(new UiSelector().textContains("Cambridge"));

        if(!viewSwitcher.exists()) { // Version 10.19.1
            cambridgeTextView.dragTo(0,getDisplayHeight(), 40);
        } 

        // Pinch to zoom, scroll around
        UiObject mapContainer = mDevice.findObject(new UiSelector().resourceId(packageID + "mainmap_container"));
        uiDeviceSwipeDown(100);
        sleep(1);
        uiDeviceSwipeUp(200);
        sleep(1);
        uiObjectVertPinchIn(mapContainer, 100, 25);
        sleep(1);
        uiDeviceSwipeLeft(100);
        uiDeviceSwipeUp(100);
        uiObjectVertPinchOut(mapContainer, 100, 50);
        sleep(3);

        // On newer versions swipe the location info tab back up
        if(!viewSwitcher.exists()) { // Version 10.19.1
            cambridgeTextView.dragTo(0, getDisplayCentreHeight(), 40);
        }

        // Get directions from Cambridge train station to Corpus Christi college
        getDirectionsFromLocation();
        search("Corpus Christi, Cambridge, England", "directions_startpoint_textbox");
        selectSearchResultContaining("Corpus Christi College");
        sleep(3);

        // View the steps for the route
        viewRouteSteps();

        // Preview the first three steps of the route
        previewRoute();
        for (int i = 0; i < 3; ++i) {
            previewNextRouteStep();
            sleep(1);
        }

        // Return to the normal map view
        pressBack();
        pressBack();
        pressBack();
    }

    public void search(String query, String box) throws Exception {
        UiObject searchBox = mDevice.findObject(new UiSelector().resourceId(packageID + box)
                                                                .className("android.widget.EditText"));
        if (!searchBox.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find search box.");
        }
        searchBox.click();

        UiObject searchText = mDevice.findObject(new UiSelector().resourceId(packageID + "search_omnibox_edit_text")
                                                                 .className("android.widget.EditText"));
        searchText.click();
        searchText.setText(query);
    }

    public void selectSearchResultContaining(String str) throws Exception {
        UiObject match = mDevice.findObject(new UiSelector().textContains(str)
                                                            .className("android.widget.TextView"));
        if (!match.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find search result containing \"" + str + "\".");
        }
        match.click();
    }

    public void getDirectionsFromLocation() throws Exception {
        UiObject directions = mDevice.findObject(new UiSelector().resourceId(packageID + "placepage_directions_button"));
        if (!directions.exists()){
            directions = mDevice.findObject(new UiSelector().textContains("DIRECTIONS"));
        }
        directions.clickAndWaitForNewWindow(uiAutoTimeout);
    }

    public void dismissLocationTutorial() throws Exception {
        UiObject gotItButton = mDevice.findObject(new UiSelector().resourceId(packageID + "tutorial_pull_up_got_it"));
        if (gotItButton.waitForExists(uiAutoTimeout)) {
            gotItButton.clickAndWaitForNewWindow(uiAutoTimeout);
        }
        sleep(3);
    }

    public void viewRouteSteps() throws Exception {
        UiObject steps = mDevice.findObject(new UiSelector().textContains("STEPS & MORE")
                                                            .className("android.widget.TextView"));
        if (steps.exists()){
            steps.clickAndWaitForNewWindow(uiAutoTimeout);
        }
    }

    public void previewRoute() throws Exception {
        UiObject preview = mDevice.findObject(new UiSelector().resourceId(packageID + "start_button"));
        preview.clickAndWaitForNewWindow(uiAutoTimeout);
    }

    public void previewNextRouteStep() throws Exception {
        UiObject next = getUiObjectByDescription("Show next", "android.widget.ImageView");
        next.click();
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }

    public void runApplicationSetup() throws Exception {
        // Dismiss 'Get the most from Google Maps' splash screen
        UiObject skipButton;
        skipButton = mDevice.findObject(new UiSelector().textContains("Skip")
                                                        .className("android.widget.Button"));
        if (skipButton.exists()) {
            skipButton.clickAndWaitForNewWindow(uiAutoTimeout);
        }

        // Dismiss a dialog regarding real-time traffic updates
        UiObject turnOffButton;
        turnOffButton = mDevice.findObject(new UiSelector().textContains("TURN OFF")
                                                           .className("android.widget.TextView"));
        if (turnOffButton.exists()) {
            turnOffButton.clickAndWaitForNewWindow(uiAutoTimeout);
        }

        // Dismiss a dialog regarding the availability of location services
        UiObject cancelButton;
        cancelButton = mDevice.findObject(new UiSelector().textContains("CANCEL")
                                                          .className("android.widget.Button"));
        if (cancelButton.exists()) {
            cancelButton.clickAndWaitForNewWindow(uiAutoTimeout);
        }
    }
}
