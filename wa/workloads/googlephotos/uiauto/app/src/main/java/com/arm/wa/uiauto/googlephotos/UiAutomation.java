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

package com.arm.wa.uiauto.googlephotos;

import android.content.Intent;
import android.graphics.Rect;
import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import com.arm.wa.uiauto.UxPerfUiAutomation.GestureTestParams;
import com.arm.wa.uiauto.UxPerfUiAutomation.GestureType;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.UiAutoUtils;
import com.arm.wa.uiauto.ActionLogger;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_DESC;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_ID;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_TEXT;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation implements ApplaunchInterface {

    private long viewTimeout = TimeUnit.SECONDS.toMillis(10);

    protected Bundle parameters;
    protected String packageID;

    @Before
    public void initialize(){
        parameters = getParams();
        packageID = getPackageID(parameters);
    }

    @Test
    public void setup() throws Exception{
        runApplicationSetup();
    }

    @Test
    public void runWorkload() throws Exception {
        selectGalleryFolder("wa-1");
        selectFirstImage();
        gesturesTest();
        navigateUp();

        selectGalleryFolder("wa-2");
        selectFirstImage();
        editPhotoColorTest();
        closeAndReturn(true);
        navigateUp();

        selectGalleryFolder("wa-3");
        selectFirstImage();
        cropPhotoTest();
        closeAndReturn(true);
        navigateUp();

        selectGalleryFolder("wa-4");
        selectFirstImage();
        rotatePhotoTest();
        closeAndReturn(true);
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }


     // Get application parameters and clear the initial run dialogues of the application launch.
    public void runApplicationSetup() throws Exception {
        sleep(5); // Pause while splash screen loads
        setScreenOrientation(ScreenOrientation.NATURAL);

        // Clear the initial run dialogues of the application launch.
        dismissWelcomeView();
        closePromotionPopUp();
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
        UiObject launchEndObject = mDevice.findObject(new UiSelector().textContains("Photos")
                                          .className("android.widget.TextView"));
        return launchEndObject;
    }

    public void dismissWelcomeView() throws Exception {
        // Click through the first two pages and make sure that we don't sign
        // in to our google account. This ensures the same set of photographs
        // are placed in the camera directory for each run.
        UiObject getStartedButton =
            mDevice.findObject(new UiSelector().textContains("Get started")
                                               .className("android.widget.Button"));
        if (getStartedButton.waitForExists(viewTimeout)) {
            getStartedButton.click();
        }

        // A network connection is not required for this workload. However,
        // when the Google Photos app is invoked from the multiapp workload a
        // connection is required for sharing content. Handle the different UI
        // pathways when dismissing welcome views here.
        UiObject doNotSignInButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "dont_sign_in_button"));

        UiObject accountName =
            mDevice.findObject(new UiSelector().resourceId(packageID + "name")
                                               .className("android.widget.TextView"));
        if (doNotSignInButton.exists()) {
            doNotSignInButton.click();
        }
        else if (accountName.exists()) {
            accountName.click();
            clickUiObject(BY_TEXT, "Use without an account", "android.widget.TextView", true);
        }
        //Some devices get popup asking for confirmation to not use backup.
        UiObject keepBackupOff =
        mDevice.findObject(new UiSelector().textContains("Keep Off")
                                           .className("android.widget.Button"));
        if (keepBackupOff.exists()){
            keepBackupOff.click();
        }

        UiObject nextButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "next_button")
                                               .className("android.widget.ImageView"));
        if (nextButton.exists()) {
            nextButton.clickAndWaitForNewWindow();
        }
    }

    public void closePromotionPopUp() throws Exception {
        UiObject promoCloseButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "promo_close_button"));
        if (promoCloseButton.exists()) {
            promoCloseButton.click();
        }
    }

    // Helper to click on the first image
    public void selectFirstImage() throws Exception {
        UiObject photo =
            mDevice.findObject(new UiSelector().resourceId(packageID + "recycler_view")
                                               .childSelector(new UiSelector()
                                               .index(1)));
        if (photo.exists()) {
            photo.click();
        } else {
            // On some versions of the app a non-zero index is used for the
            // photographs position while on other versions a zero index is used.
            // Try both possiblities before throwing an exception.
            photo =
                mDevice.findObject(new UiSelector().resourceId(packageID + "recycler_view")
                                                   .childSelector(new UiSelector()
                                                   .index(0)));
            photo.click();
        }
    }

    // Helper that accepts, closes and navigates back to application home screen after an edit operation.
    // dontsave - True will discard the image. False will save the image
    public void closeAndReturn(final boolean dontsave) throws Exception {
        long timeout =  TimeUnit.SECONDS.toMillis(3);

        UiObject accept =
            mDevice.findObject(new UiSelector().description("Accept"));
        UiObject done =
            mDevice.findObject(new UiSelector().resourceId(packageID + "cpe_save_button")
                                               .textContains("Done"));

        // On some edit operations we can either confirm an edit with "Accept", "DONE" or neither.
        if (accept.waitForExists(timeout)) {
            accept.click();
        } else if (done.waitForExists(timeout)) {
            done.click();
        }

        if (dontsave) {
            clickUiObject(BY_DESC, "Close editor", "android.widget.ImageView");

            UiObject discard = getUiObjectByText("DISCARD", "android.widget.Button");
            discard.waitForExists(viewTimeout);
            discard.click();
        } else {
            UiObject save = getUiObjectByText("SAVE", "android.widget.TextView");
            save.waitForExists(viewTimeout);
            save.click();
        }
    }

    public void navigateUp() throws Exception {
        // Navigate up to go to folder
        UiObject navigateUpButton =
            clickUiObject(BY_DESC, "Navigate Up", "android.widget.ImageButton", true);
        // Navigate up again to go to gallery - if it exists
        if (navigateUpButton.exists()) {
            navigateUpButton.clickAndWaitForNewWindow();
        }
    }

    private void gesturesTest() throws Exception {
        String testTag = "gesture";

        // Perform a range of swipe tests while browsing photo gallery
        LinkedHashMap<String, GestureTestParams> testParams = new LinkedHashMap<String, GestureTestParams>();
        testParams.put("pinch_out", new GestureTestParams(GestureType.PINCH, PinchType.OUT, 100, 50));
        testParams.put("pinch_in", new GestureTestParams(GestureType.PINCH, PinchType.IN, 100, 50));

        Iterator<Entry<String, GestureTestParams>> it = testParams.entrySet().iterator();

        while (it.hasNext()) {
            Map.Entry<String, GestureTestParams> pair = it.next();
            GestureType type = pair.getValue().gestureType;
            PinchType pinch = pair.getValue().pinchType;
            int steps = pair.getValue().steps;
            int percent = pair.getValue().percent;

            UiObject view =
                mDevice.findObject(new UiSelector().enabled(true));
            if (!view.waitForExists(viewTimeout)) {
                throw new UiObjectNotFoundException("Could not find \"photo view\".");
            }

            String runName = String.format(testTag + "_" + pair.getKey());
            ActionLogger logger = new ActionLogger(runName, parameters);
            logger.start();

            switch (type) {
                case PINCH:
                    uiObjectVertPinch(view, pinch, steps, percent);
                    break;
                default:
                    break;
            }

            logger.stop();
        }
    }

    public enum Position { LEFT, RIGHT, CENTRE };

    private class PositionPair {
        private Position start;
        private Position end;

        PositionPair(final Position start, final Position end) {
            this.start = start;
            this.end = end;
        }
    }

    private void editPhotoColorTest() throws Exception {
        long timeout =  TimeUnit.SECONDS.toMillis(3);
        // To improve travel accuracy perform the slide bar operation slowly
        final int steps = 100;

        String testTag = "edit";

        // Perform a range of swipe tests while browsing photo gallery
        LinkedHashMap<String, PositionPair> testParams = new LinkedHashMap<String, PositionPair>();
        testParams.put("color_increment", new PositionPair(Position.CENTRE, Position.RIGHT));
        testParams.put("color_reset", new PositionPair(Position.RIGHT, Position.CENTRE));
        testParams.put("color_decrement", new PositionPair(Position.CENTRE, Position.LEFT));

        Iterator<Entry<String, PositionPair>> it = testParams.entrySet().iterator();

        clickUiObject(BY_ID, packageID + "edit", "android.widget.ImageView");

        // Manage potential different spelling of UI element
        UiObject editCol =
            mDevice.findObject(new UiSelector().textMatches("Colou?r"));
        if (editCol.waitForExists(timeout)) {
            editCol.click();
        } else {
            UiObject adjustTool =
                mDevice.findObject(new UiSelector().resourceId(packageID + "cpe_adjustments_tool")
                                                   .className("android.widget.ImageView"));
            if (adjustTool.waitForExists(timeout)){
                adjustTool.click();
            } else {
                throw new UiObjectNotFoundException(String.format("Could not find Color/Colour adjustment"));
            }
        }

        UiObject seekBar =
            mDevice.findObject(new UiSelector().resourceId(packageID + "cpe_strength_seek_bar")
                                               .className("android.widget.SeekBar"));
        if (!(seekBar.exists())){
            seekBar =
            mDevice.findObject(new UiSelector().resourceIdMatches(".*/cpe_adjustments_section_slider")
                                               .className("android.widget.SeekBar").descriptionMatches("Colou?r"));
        }

        while (it.hasNext()) {
            Map.Entry<String, PositionPair> pair = it.next();
            Position start = pair.getValue().start;
            Position end = pair.getValue().end;

            String runName = String.format(testTag + "_" + pair.getKey());
            ActionLogger logger = new ActionLogger(runName, parameters);

            logger.start();
            seekBarTest(seekBar, start, end, steps);
            logger.stop();
        }
    }

    private void cropPhotoTest() throws Exception {
        String testTag = "crop";

        // To improve travel accuracy perform the slide bar operation slowly
        final int steps = 100;

        // Perform a range of swipe tests while browsing photo gallery
        LinkedHashMap<String, Position> testParams = new LinkedHashMap<String, Position>();
        testParams.put("tilt_positive", Position.LEFT);
        testParams.put("tilt_reset", Position.RIGHT);
        testParams.put("tilt_negative", Position.RIGHT);

        Iterator<Entry<String, Position>> it = testParams.entrySet().iterator();

        clickUiObject(BY_ID, packageID + "edit", "android.widget.ImageView");
        clickUiObject(BY_ID, packageID + "cpe_crop_tool", "android.widget.ImageView");

        UiObject straightenSlider =
            getUiObjectByResourceId(packageID + "cpe_straighten_slider");

        while (it.hasNext()) {
            Map.Entry<String, Position> pair = it.next();
            Position pos = pair.getValue();

            String runName = String.format(testTag + "_" + pair.getKey());
            ActionLogger logger = new ActionLogger(runName, parameters);

            logger.start();
            slideBarTest(straightenSlider, pos, steps);
            logger.stop();
        }
    }

    private void rotatePhotoTest() throws Exception {
        String testTag = "rotate";

        String[] subTests = {"90", "180", "270"};

        clickUiObject(BY_ID, packageID + "edit", "android.widget.ImageView");
        clickUiObject(BY_ID, packageID + "cpe_crop_tool", "android.widget.ImageView");

        UiObject rotate =
            getUiObjectByResourceId(packageID + "cpe_rotate_90");

        for (String subTest : subTests) {
            String runName = String.format(testTag + "_" + subTest);
            ActionLogger logger = new ActionLogger(runName, parameters);

            logger.start();
            rotate.click();
            logger.stop();
        }
    }

    // Helper to slide the seekbar during photo edit.
    private void seekBarTest(final UiObject view, final Position start, final Position end, final int steps) throws Exception {
        final int SWIPE_MARGIN_LIMIT = 5;
        Rect rect = view.getVisibleBounds();
        int startX, endX;

        switch (start) {
            case CENTRE:
                startX = rect.centerX();
                break;
            case LEFT:
                startX = rect.left + SWIPE_MARGIN_LIMIT;
                break;
            case RIGHT:
                startX = rect.right - SWIPE_MARGIN_LIMIT;
                break;
            default:
                startX = 0;
                break;
        }

        switch (end) {
            case CENTRE:
                endX = rect.centerX();
                break;
            case LEFT:
                endX = rect.left + SWIPE_MARGIN_LIMIT;
                break;
            case RIGHT:
                endX = rect.right - SWIPE_MARGIN_LIMIT;
                break;
            default:
                endX = 0;
                break;
        }

        mDevice.drag(startX, rect.centerY(), endX, rect.centerY(), steps);
    }

    // Helper to slide the slidebar during photo edit.
    private void slideBarTest(final UiObject view, final Position pos, final int steps) throws Exception {
        final int SWIPE_MARGIN_LIMIT = 5;
        Rect rect = view.getBounds();

        switch (pos) {
            case LEFT:
                mDevice.drag(rect.left + SWIPE_MARGIN_LIMIT, rect.centerY(),
                             rect.left + (rect.width() / 4), rect.centerY(),
                             steps);
                break;
            case RIGHT:
                mDevice.drag(rect.right - SWIPE_MARGIN_LIMIT, rect.centerY(),
                             rect.right - (rect.width() / 4), rect.centerY(),
                             steps);
                break;
            default:
                break;
        }
    }
}
