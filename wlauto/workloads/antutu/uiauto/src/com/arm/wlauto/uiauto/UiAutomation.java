/*    Copyright 2013-2015 ARM Limited
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


package com.arm.wlauto.uiauto.antutu;

import java.util.Set;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.core.UiCollection;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "antutu";
    public static String TestButton5 = "com.antutu.ABenchMark:id/start_test_region";
    public static String TestButton6 = "com.antutu.ABenchMark:id/start_test_text";
    private static int initialTimeoutSeconds = 20;

    public void runUiAutomation() throws Exception{
        Bundle parameters = getParams();

        String version = parameters.getString("version");
        boolean enableSdTests = Boolean.parseBoolean(parameters.getString("enable_sd_tests"));

        int times = Integer.parseInt(parameters.getString("times"));
        if (times < 1) {
                times = 1;
        }

        if (version.equals("3.3.2")) { // version earlier than 4.0.3
            dismissReleaseNotesDialogIfNecessary();
            if(!enableSdTests){
               disableSdCardTests();
            }
            hitStart();
            waitForAndViewResults();
        }
        else {
            int iteration = 0;
            dismissNewVersionNotificationIfNecessary();
            while (true) {
                    if(version.equals("6.0.1"))
                        hitTestButtonVersion5(TestButton6);
                    else if (version.equals("5.3.0")) {
                        hitTestButton();
                        hitTestButtonVersion5(TestButton5);
                    }
                    else if (version.equals("4.0.3")) {
                        hitTestButton();
                        hitTestButton();
                    }
                    else
                        hitTestButton();

                    if(version.equals("6.0.1"))
                    {
                        waitForVersion6Results();
                        extractResults6();
                    }
                    else
                    {
                        waitForVersion4Results();
                        viewDetails();
                        extractResults();
                    }

                    iteration++;
                    if (iteration >= times) {
                        break;
                    }

                    returnToTestScreen(version);
                    dismissRateDialogIfNecessary();
                    testAgain();
            }
        }

        Bundle status = new Bundle();
        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

    public boolean dismissNewVersionNotificationIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject closeButton = new UiObject(selector.text("Cancel"));
        if (closeButton.waitForExists(TimeUnit.SECONDS.toMillis(initialTimeoutSeconds))) {
            closeButton.click();
            sleep(1); // diaglog dismissal
            return true;
        } else {
            return false;
        }
    }

    public boolean dismissReleaseNotesDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject closeButton = new UiObject(selector.text("Close"));
        if (closeButton.waitForExists(TimeUnit.SECONDS.toMillis(initialTimeoutSeconds))) {
            closeButton.click();
            sleep(1); // diaglog dismissal
            return true;
        } else {
            return false;
        }
    }

    public boolean dismissRateDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject closeButton = new UiObject(selector.text("NOT NOW"));
        boolean dismissed = false;
        // Sometimes, dismissing the dialog the first time does not work properly --
        // it starts to disappear but is then immediately re-created; so may need to
        // dismiss it as long as keeps popping up.
        while (closeButton.waitForExists(2)) {
            closeButton.click();
            sleep(1); // diaglog dismissal
            dismissed = true;
        }
        return dismissed;
    }

    public void hitTestButton() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject test = new UiObject(selector.text("Test")
                                             .className("android.widget.Button"));
        test.waitForExists(initialTimeoutSeconds);
        test.click();
        sleep(1); // possible tab transtion
    }

   /* In version 5 of antutu, the test has been changed from a button widget to a textview */

   public void hitTestButtonVersion5(String id) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject test = new UiObject(selector.resourceId(id)
                                             .className("android.widget.TextView"));
        test.waitForExists(initialTimeoutSeconds);
        test.click();
        sleep(1); // possible tab transtion
    }


    public void hitTest() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject test = new UiObject(selector.text("Test"));
        test.click();
        sleep(1); // possible tab transtion
    }

    public void disableSdCardTests() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject custom = new UiObject(selector.textContains("Custom"));
        custom.click();
        sleep(1); // tab transition

        UiObject sdCardButton = new UiObject(selector.text("SD card IO"));
        sdCardButton.click();
    }

    public void hitStart() throws Exception {
        UiSelector selector = new UiSelector();
        Log.v(TAG, "Start the test");
        UiObject startButton = new UiObject(selector.text("Start Test")
                                                    .className("android.widget.Button"));
        startButton.click();
    }

    public void waitForVersion4Results() throws Exception {
        // The observed behaviour seems to vary between devices. On some platforms,
        // the benchmark terminates in the barchart screen; on others, it terminates in
        // details screen. So we have to wait for either and then act appropriatesl (on the barchart
        // screen a back button press is required to get to the details screen.
        UiSelector selector = new UiSelector();
        UiObject barChart = new UiObject(new UiSelector().className("android.widget.TextView")
                                                         .text("Bar Chart"));
        UiObject detailsButton = new UiObject(new UiSelector().className("android.widget.Button")
                                                              .text("Details"));
        for (int i = 0; i < 60; i++) {
            if (detailsButton.exists() || barChart.exists()) {
                break;
            }
            sleep(5);
        }

        if (barChart.exists()) {
            getUiDevice().pressBack();
        }
    }

    public void waitForVersion6Results() throws Exception {
        UiObject qrText = new UiObject(new UiSelector().className("android.widget.TextView")
                                                       .text("QRCode of result"));
        for (int i = 0; i < 120; i++) {
            if (qrText.exists()) {
                break;
            }
            sleep(5);
        }
    }

    public void viewDetails() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject detailsButton = new UiObject(new UiSelector().className("android.widget.Button")
                                                              .text("Details"));
        detailsButton.clickAndWaitForNewWindow();
    }

    public void extractResults6() throws Exception {
        //Overal result
        UiObject result = new UiObject(new UiSelector().resourceId("com.antutu.ABenchMark:id/tv_score_name"));
        if (result.exists()) {
            Log.v(TAG, String.format("ANTUTU RESULT: Overall Score: %s", result.getText()));
        }

        // individual scores
        extractSectionResults6("3d");
        extractSectionResults6("ux");
        extractSectionResults6("cpu");
        extractSectionResults6("ram");
    }

    public void extractSectionResults6(String section) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject resultLayout = new UiObject(selector.resourceId("com.antutu.ABenchMark:id/hcf_" + section));
        UiObject result = resultLayout.getChild(selector.resourceId("com.antutu.ABenchMark:id/tv_score_value"));

        if (result.exists()) {
            Log.v(TAG, String.format("ANTUTU RESULT: %s Score: %s", section, result.getText()));
        }
    }

    public void extractResults() throws Exception {
        extractOverallResult();
        extractSectionResults();
    }

    public void extractOverallResult() throws Exception {
        UiSelector selector = new UiSelector();
        UiSelector resultTextSelector = selector.className("android.widget.TextView").index(0);
        UiSelector relativeLayoutSelector = selector.className("android.widget.RelativeLayout").index(1);
        UiObject result = new UiObject(selector.className("android.widget.LinearLayout")
                                               .childSelector(relativeLayoutSelector)
                                               .childSelector(resultTextSelector));
        if (result.exists()) {
            Log.v(TAG, String.format("ANTUTU RESULT: Overall Score: %s", result.getText()));
        }
    }

    public void extractSectionResults() throws Exception {
        UiSelector selector = new UiSelector();
        Set<String> processedMetrics = new HashSet<String>();

        actuallyExtractSectionResults(processedMetrics);
        UiScrollable resultsList = new UiScrollable(selector.className("android.widget.ScrollView"));
        // Note: there is an assumption here that the entire results list fits on at most
        //       two screens on the device. Given then number of entries in the current
        //       antutu verion and the devices we're dealing with, this is a reasonable
        //       assumption. But if this changes, this will need to be adapted to scroll more
        //       slowly.
        resultsList.scrollToEnd(10);
        actuallyExtractSectionResults(processedMetrics);
    }

    public void actuallyExtractSectionResults(Set<String> processedMetrics) throws Exception {
        UiSelector selector = new UiSelector();

        for (int i = 1; i < 8; i += 2) {
            UiObject table = new UiObject(selector.className("android.widget.TableLayout").index(i));
            for (int j = 0; j < 3; j += 2) {
                UiObject row = table.getChild(selector.className("android.widget.TableRow").index(j));
                UiObject metric =  row.getChild(selector.className("android.widget.TextView").index(0));
                UiObject value =  row.getChild(selector.className("android.widget.TextView").index(1));

                if (metric.exists() && value.exists()) {
                    String metricText = metric.getText();
                    if (!processedMetrics.contains(metricText)) {
                        Log.v(TAG, String.format("ANTUTU RESULT: %s %s", metric.getText(), value.getText()));
                        processedMetrics.add(metricText);
                    }
                }
            }
        }
    }

    public void returnToTestScreen(String version) throws Exception {
        getUiDevice().pressBack();
        if (version.equals("5.3.0"))
        {
            UiSelector selector = new UiSelector();
            UiObject detailsButton = new UiObject(new UiSelector().className("android.widget.Button")
                                                                  .text("Details"));
            sleep(1);
            getUiDevice().pressBack();
        }
    }

    public void testAgain() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject retestButton = new UiObject(selector.text("Test Again")
                                                     .className("android.widget.Button"));
        if (!retestButton.waitForExists(TimeUnit.SECONDS.toMillis(2))) {
            getUiDevice().pressBack();
            retestButton.waitForExists(TimeUnit.SECONDS.toMillis(2));
        }
        retestButton.clickAndWaitForNewWindow();
    }

    public void waitForAndViewResults() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject submitTextView = new UiObject(selector.text("Submit Scores")
                                                       .className("android.widget.TextView"));
        UiObject detailTextView = new UiObject(selector.text("Detailed Scores")
                                                       .className("android.widget.TextView"));
        UiObject commentTextView = new UiObject(selector.text("User comment")
                                                        .className("android.widget.TextView"));
        boolean foundResults = false;
        for (int i = 0; i < 60; i++) {
            if (detailTextView.exists() || submitTextView.exists() || commentTextView.exists()) {
                foundResults = true;
                break;
            }
            sleep(5);
        }

        if (!foundResults) {
                throw new UiObjectNotFoundException("Did not see AnTuTu results screen.");
        }

        if (commentTextView.exists()) {
            getUiDevice().pressBack();
        }
        // Yes, sometimes, it needs to be done twice...
        if (commentTextView.exists()) {
            getUiDevice().pressBack();
        }

        if (detailTextView.exists()) {
            detailTextView.click();
            sleep(1); // tab transition

            UiObject testTextView = new UiObject(selector.text("Test")
                                                    .className("android.widget.TextView"));
            if (testTextView.exists()) {
            testTextView.click();
            sleep(1); // tab transition
            }

            UiObject scoresTextView = new UiObject(selector.text("Scores")
                                                    .className("android.widget.TextView"));
            if (scoresTextView.exists()) {
            scoresTextView.click();
            sleep(1); // tab transition
            }
        }
    }
}
