/*    Copyright 2014-2015 ARM Limited
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


package com.arm.wlauto.uiauto.vellamo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.core.UiDevice;
import com.android.uiautomator.core.UiWatcher;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "vellamo";
    public static ArrayList<String> scores = new ArrayList();
    public static Boolean wasError = false;

    public void runUiAutomation() throws Exception {
        Bundle parameters = getParams();
        String version = parameters.getString("version");
        Boolean browser = Boolean.parseBoolean(parameters.getString("browser"));
        Boolean metal = Boolean.parseBoolean(parameters.getString("metal"));
        Boolean multicore = Boolean.parseBoolean(parameters.getString("multicore"));
        Integer browserToUse = Integer.parseInt(parameters.getString("browserToUse")) - 1;

        dismissEULA();

        if (version.equals("2.0.3")) {
            dissmissWelcomebanner();
            startTest();
            dismissNetworkConnectionDialogIfNecessary();
            dismissExplanationDialogIfNecessary();
            waitForTestCompletion(15 * 60, "com.quicinc.vellamo:id/act_ba_results_btn_no");
            getScore("html5", "com.quicinc.vellamo:id/act_ba_results_img_0");
            getScore("metal", "com.quicinc.vellamo:id/act_ba_results_img_1");
        }

        else {
            dismissLetsRoll();
            if (browser) {
                startBrowserTest(browserToUse);
                proccessTest("Browser");
            }
            if (multicore) {
                startTestV3(1);
                proccessTest("Multicore");

            }
            if (metal) {
                startTestV3(2);
                proccessTest("Metal");
            }
        }
        for(String result : scores){
            Log.v(TAG, String.format("VELLAMO RESULT: %s", result));
        }
        if (wasError) Log.v("vellamoWatcher", "VELLAMO ERROR: Something crashed while running browser benchmark");
    }

    public void startTest() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runButton = new UiObject(selector.textContains("Run All Chapters"));

        if (!runButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            UiObject pager = new UiObject(selector.className("android.support.v4.view.ViewPager"));
            pager.swipeLeft(2);
            if (!runButton.exists()) {
                throw new UiObjectNotFoundException("Could not find \"Run All Chapters\" button.");
            }
        }
        runButton.click();
    }

    public void startBrowserTest(int browserToUse) throws Exception {
        //Ensure chrome is selected as "browser" fails to run the benchmark
        UiSelector selector = new UiSelector();
        UiObject browserToUseButton = new UiObject(selector.className("android.widget.ImageButton")
                                               .longClickable(true).instance(browserToUse));
        UiObject browserButton = new UiObject(selector.className("android.widget.ImageButton")
                                               .longClickable(true).selected(true));
        //Disable browsers
        while(browserButton.exists()) browserButton.click();
        if (browserToUseButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (browserToUseButton.exists()) {
                browserToUseButton.click();
            }
        }

        //enable a watcher to dismiss browser dialogs
        UiWatcher stoppedWorkingDialogWatcher = new UiWatcher() {
            @Override
            public boolean checkForCondition() {
                UiObject stoppedWorkingDialog = new UiObject(new UiSelector().textStartsWith("Unfortunately"));
                if(stoppedWorkingDialog.exists()){
                    wasError = true;
                    UiObject okButton = new UiObject(new UiSelector().className("android.widget.Button").text("OK"));
                    try {
                        okButton.click();
                    } catch (UiObjectNotFoundException e) {
                        // TODO Auto-generated catch block
                        e.printStackTrace();
                    }
                    return (stoppedWorkingDialog.waitUntilGone(25000));
                }
                return false;
            }
        };
        // Register watcher
        UiDevice.getInstance().registerWatcher("stoppedWorkingDialogWatcher", stoppedWorkingDialogWatcher);

        // Run watcher
        UiDevice.getInstance().runWatchers();

        startTestV3(0);
    }

    public void startTestV3(int run) throws Exception {
        UiSelector selector = new UiSelector();

        UiObject thirdRunButton = new UiObject(selector.resourceId("com.quicinc.vellamo:id/card_launcher_run_button").instance(run));
        if (!thirdRunButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!thirdRunButton.exists()) {
                throw new UiObjectNotFoundException("Could not find three \"Run\" buttons.");
            }
        }

        //Run benchmarks
        UiObject runButton = new UiObject(selector.resourceId("com.quicinc.vellamo:id/card_launcher_run_button").instance(run));
        if (!runButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!runButton.exists()) {
                throw new UiObjectNotFoundException("Could not find correct \"Run\" button.");
            }
        }
        runButton.click();

        //Skip tutorial screens
        UiObject swipeScreen = new UiObject(selector.textContains("Swipe left to continue"));
        if (!swipeScreen.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!swipeScreen.exists()) {
                throw new UiObjectNotFoundException("Could not find \"Swipe screen\".");
            }
        }
        sleep(1);
        swipeScreen.swipeLeft(2);
        sleep(1);
        swipeScreen.swipeLeft(2);

    }

    public void proccessTest(String metric) throws Exception{
        waitForTestCompletion(15 * 60, "com.quicinc.vellamo:id/button_no");

        //Remove watcher
        UiDevice.getInstance().removeWatcher("stoppedWorkingDialogWatcher");

        getScore(metric, "com.quicinc.vellamo:id/card_score_score");
        getUiDevice().pressBack();
        getUiDevice().pressBack();
        getUiDevice().pressBack();
    }

    public void getScore(String metric, String resourceID) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject score = new UiObject(selector.resourceId(resourceID));
        if (!score.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!score.exists()) {
                throw new UiObjectNotFoundException("Could not find score on screen.");
            }
        }
        scores.add(metric + " " + score.getText().trim());
    }

    public void waitForTestCompletion(int timeout, String resourceID) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject resultsNoButton = new UiObject(selector.resourceId(resourceID));
        if (!resultsNoButton.waitForExists(TimeUnit.SECONDS.toMillis(timeout))) {
            throw new UiObjectNotFoundException("Did not see results screen.");
        }

    }

    public void dismissEULA() throws Exception {
        UiSelector selector = new UiSelector();
        waitText("Vellamo EULA");
        UiObject acceptButton = new UiObject(selector.text("Accept")
                                                     .className("android.widget.Button"));
        if (acceptButton.exists()) {
            acceptButton.click();
        }
    }

    public void dissmissWelcomebanner() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject welcomeBanner = new UiObject(selector.textContains("WELCOME"));
        if (welcomeBanner.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            UiObject pager = new UiObject(selector.className("android.support.v4.view.ViewPager"));
            pager.swipeLeft(2);
            pager.swipeLeft(2);
        }
    }

    public void dismissLetsRoll() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject letsRollButton = new UiObject(selector.className("android.widget.Button")
                                                       .textContains("Let's Roll"));
        if (!letsRollButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!letsRollButton.exists()) {
                throw new UiObjectNotFoundException("Could not find \"Let's Roll\" button.");
            }
        }
        letsRollButton.click();
    }

    public void dismissNetworkConnectionDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject dialog = new UiObject(selector.className("android.widget.TextView")
                                               .textContains("No Network Connection"));
        if (dialog.exists()) {
            UiObject yesButton = new UiObject(selector.className("android.widget.Button")
                                                      .text("Yes"));
            yesButton.click();
        }
    }

    public void dismissExplanationDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject dialog = new UiObject(selector.className("android.widget.TextView")
                                               .textContains("Benchmarks Explanation"));
        if (dialog.exists()) {
            UiObject noButton = new UiObject(selector.className("android.widget.Button")
                                                     .text("No"));
            noButton.click();
        }
    }
}
