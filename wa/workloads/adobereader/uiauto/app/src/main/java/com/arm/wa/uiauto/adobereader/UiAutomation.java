package com.arm.wa.uiauto.adobereader;

/*    Copyright 2014-2016 ARM Limited
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


import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.UxPerfUiAutomation.GestureTestParams;
import com.arm.wa.uiauto.UxPerfUiAutomation.GestureType;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ActionLogger;
import com.arm.wa.uiauto.UiAutoUtils;


import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_DESC;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_ID;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_TEXT;


@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation implements ApplaunchInterface {

    private long networkTimeout =  TimeUnit.SECONDS.toMillis(20);
    private long searchTimeout =  TimeUnit.SECONDS.toMillis(20);

    protected Bundle parameters;
    protected String packageID;
    protected String filename;
    protected String[] searchStrings;


    @Before
    public void initialize(){
        parameters = getParams();
        packageID = getPackageID(parameters);
        filename = parameters.getString("filename");
        searchStrings = parameters.getStringArray("search_string_list");
    }

    @Test
    public void setup() throws Exception {
        runApplicationSetup();
    }

    @Test
    public void runWorkload() throws Exception {
        openFile(filename);
        gesturesTest();
        searchPdfTest(searchStrings);
        exitDocument();
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }

    public void runApplicationSetup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        dismissWelcomeView();
    }

    // Returns the launch command for the application.
    public String getLaunchCommand() {
        String launch_command;
        launch_command = UiAutoUtils.createLaunchCommand(parameters);
        return launch_command;
    }

    // Pass the workload parameters, used for applaunch
    public void setWorkloadParameters(Bundle workload_parameters) {
        parameters = workload_parameters;
        packageID = getPackageID(parameters);
    }

    // Sets the UiObject that marks the end of the application launch.
    public UiObject getLaunchEndObject() {
        UiObject launchEndObject =
            mDevice.findObject(new UiSelector().textContains("RECENT")
                                               .className("android.widget.TextView"));
        return launchEndObject;
    }


    private void dismissWelcomeView() throws Exception {
        UiObject welcomeView = getUiObjectByResourceId("android:id/content",
                                                       "android.widget.FrameLayout");
        welcomeView.swipeLeft(10);
        welcomeView.swipeLeft(10);

        UiObject onboarding_finish_button =
            mDevice.findObject(new UiSelector().resourceId("com.adobe.reader:id/onboarding_finish_button"));

        if (!onboarding_finish_button.exists()) {
            welcomeView.swipeLeft(10);
        }

        if (onboarding_finish_button.exists()) {
            clickUiObject(BY_ID, packageID + "onboarding_finish_button", "android.widget.Button");
        }

        // Deal with popup dialog message promoting Dropbox access
        UiObject dropBoxDialog =
                mDevice.findObject(new UiSelector().text("Now you can access your Dropbox files.")
                        .className("android.widget.TextView"));
        if (dropBoxDialog.exists()) {
            clickUiObject(BY_TEXT, "Remind Me Later", "android.widget.Button");
        }

        // Also deal with the Dropbox CoachMark blue hint popup
        UiObject dropBoxcoachMark =
                mDevice.findObject(new UiSelector().description("CoachMark")
                                                   .className("android.widget.LinearLayout"));
        if (dropBoxcoachMark.exists()) {
            tapDisplayCentre();
        }

        UiObject actionBarTitle = mDevice.findObject(new UiSelector().textContains("My Documents")
                                                            .className("android.widget.TextView"));
        actionBarTitle.waitForExists(uiAutoTimeout);
    }

    private void openFile(final String filename) throws Exception {
        String testTag = "open_document";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        // Select the local files list from the My Documents view
        clickUiObject(BY_TEXT, "LOCAL", "android.widget.TextView");
        UiObject directoryPath =
                mDevice.findObject(new UiSelector().resourceId(packageID + "directoryPath"));
        if (!directoryPath.waitForExists(TimeUnit.SECONDS.toMillis(60))) {
            throw new UiObjectNotFoundException("Could not find any local files");
        }

        // Click the button to search from the present file list view
        UiObject searchButton =
                mDevice.findObject(new UiSelector().resourceId(packageID + "split_pane_search"));
        if (!searchButton.waitForExists(TimeUnit.SECONDS.toMillis(10))) {
            throw new UiObjectNotFoundException("Could not find search button");
        }
        searchButton.click();
        // Force a refresh of files before searching
        uiDeviceSwipe(Direction.DOWN, 100);
        // Repeat as first swipe is sometimes ignored.
        uiDeviceSwipe(Direction.DOWN, 100);

        // Enter search text into the file searchBox.  This will automatically filter the list.
        UiObject searchBox =
                mDevice.findObject(new UiSelector().resourceIdMatches(".*search_src_text")
                                                   .classNameMatches("android.widget.Edit.*"));

        searchBox.setText(filename);

        // Open a file from a file list view by searching for UiObjects containing the doc title.
        UiObject fileObject = getUiObjectByText(filename, "android.widget.TextView");

        logger.start();

        fileObject.clickAndWaitForNewWindow(uiAutoTimeout);
        // Wait for the doc to open by waiting for the viewPager UiObject to exist
        UiObject viewPager =
                mDevice.findObject(new UiSelector().resourceId(packageID + "viewPager"));
        if (!viewPager.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"viewPager\".");
        };

        logger.stop();
    }

    private void gesturesTest() throws Exception {
        String testTag = "gesture";

        // Perform a range of swipe tests at different speeds and on different views
        LinkedHashMap<String, GestureTestParams> testParams = new LinkedHashMap<String, GestureTestParams>();
        testParams.put("swipe_up", new GestureTestParams(GestureType.UIDEVICE_SWIPE, Direction.UP, 100));
        testParams.put("swipe_down", new GestureTestParams(GestureType.UIDEVICE_SWIPE, Direction.DOWN, 100));
        testParams.put("swipe_right", new GestureTestParams(GestureType.UIOBJECT_SWIPE, Direction.RIGHT, 50));
        testParams.put("swipe_left", new GestureTestParams(GestureType.UIOBJECT_SWIPE, Direction.LEFT, 50));
        testParams.put("pinch_out", new GestureTestParams(GestureType.PINCH, PinchType.OUT, 100, 50));
        testParams.put("pinch_in", new GestureTestParams(GestureType.PINCH, PinchType.IN, 100, 50));

        Iterator<Entry<String, GestureTestParams>> it = testParams.entrySet().iterator();

        // On some devices the first device swipe is ignored so perform it here
        // to prevent the first test gesture from being incorrectly logged
        uiDeviceSwipe(Direction.DOWN, 200);

        UiObject view =
                mDevice.findObject(new UiSelector().resourceId(packageID + "pageView"));
        if (!view.waitForExists(TimeUnit.SECONDS.toMillis(10))) {
            throw new UiObjectNotFoundException("Could not find page view");
        }

        while (it.hasNext()) {
            Entry<String, GestureTestParams> pair = it.next();
            GestureType type = pair.getValue().gestureType;
            Direction dir = pair.getValue().gestureDirection;
            PinchType pinch = pair.getValue().pinchType;
            int steps = pair.getValue().steps;
            int percent = pair.getValue().percent;

            String runName = String.format(testTag + "_" + pair.getKey());
            ActionLogger logger = new ActionLogger(runName, parameters);
            logger.start();

            switch (type) {
                case UIDEVICE_SWIPE:
                    uiDeviceSwipe(dir, steps);
                    break;
                case UIOBJECT_SWIPE:
                    uiObjectSwipe(view, dir, steps);
                    break;
                case PINCH:
                    uiObjectVertPinch(view, pinch, steps, percent);
                    break;
                default:
                    break;
            }

            logger.stop();
        }
    }

    private void searchPdfTest(final String[] searchStrings) throws Exception {
        String testTag = "search";

        // Tap the centre to bring up the menu gui
        // Sometimes the first tap wont register, so check if search appears
        // and if not, tap again before continuing
        tapDisplayCentre();
        UiObject searchIcon =
                mDevice.findObject(new UiSelector().resourceId(packageID + "document_view_search_icon"));
        if (!searchIcon.waitForExists(uiAutoTimeout)) {
            tapDisplayCentre();
        }

        if (!searchIcon.waitForExists(uiAutoTimeout)) {
            searchIcon =
                    mDevice.findObject(new UiSelector().resourceId(packageID + "document_view_search"));
            if (!searchIcon.waitForExists(uiAutoTimeout)) {
                tapDisplayCentre();
            }
        }

        for (int i = 0; i < searchStrings.length; i++) {
            String runName = String.format(testTag + "_string" + i);
            ActionLogger logger = new ActionLogger(runName, parameters);

            // Click on the search button icon and enter text in the box.  This closes the keyboard
            // so click the box again and press Enter to start the search.
            searchIcon.clickAndWaitForNewWindow();

            UiObject searchBox =
                    mDevice.findObject(new UiSelector().resourceIdMatches(".*search_src_text")
                                                       .className("android.widget.EditText"));

            searchBox.setText(searchStrings[i]);

            logger.start();

            pressEnter();

            // Check the progress bar icon.  When this disappears the search is complete.
            UiObject progressBar =
                    mDevice.findObject(new UiSelector().resourceId(packageID + "searchProgress")
                                                       .className("android.widget.ProgressBar"));
            progressBar.waitForExists(uiAutoTimeout);
            progressBar.waitUntilGone(searchTimeout);

            logger.stop();

            // Get back to the main document view by clicking twice on the close button
            UiObject searchCloseButton =
                    mDevice.findObject(new UiSelector().resourceIdMatches(".*search_close_btn")
                                                       .className("android.widget.ImageView"));
            searchCloseButton.click();

            if (searchCloseButton.exists()){
                searchCloseButton.clickAndWaitForNewWindow();
            }
            else {
                UiObject searchBackButton = getUiObjectByDescription("Collapse",
                                                                     "android.widget.ImageButton");
                searchBackButton.clickAndWaitForNewWindow();
            }
        }
    }

    private void exitDocument() throws Exception {
        // Return from the document view to the file list view by pressing home and my documents.
        UiObject homeButton =
                mDevice.findObject(new UiSelector().resourceId("android:id/home")
                        .className("android.widget.ImageView"));
        // Newer version of app have a menu button instead of home button.
        UiObject menuButton =
                mDevice.findObject(new UiSelector().description("Navigate up"));

        if (!(homeButton.exists() || menuButton.exists())){
            tapDisplayCentre();
        }

        if (homeButton.exists()){
            homeButton.click();
        }
        else if (menuButton.exists()){
            menuButton.click();
        }
        else {
            menuButton =
                    mDevice.findObject(new UiSelector().resourceIdMatches(".*up.*")
                                            .classNameMatches("android.widget.Image.*"));
            menuButton.click();
        }

        clickUiObject(BY_DESC, "My Documents", "android.widget.LinearLayout", true);

        UiObject searchBackButton =
                mDevice.findObject(new UiSelector().description("Collapse")
                                                   .className("android.widget.ImageButton"));
        if (searchBackButton.exists()){
            searchBackButton.click();
        }
        else {
            clickUiObject(BY_ID, "android:id/up", "android.widget.ImageView", true);
        }
    }
}
