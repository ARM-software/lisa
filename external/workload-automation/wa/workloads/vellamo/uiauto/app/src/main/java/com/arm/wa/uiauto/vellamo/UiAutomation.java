
/*    Copyright 2014-2017 ARM Limited
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


package com.arm.wa.uiauto.vellamo;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiWatcher;
import android.util.Log;

import com.arm.wa.uiauto.BaseUiAutomation;

import org.junit.Test;
import org.junit.Before;
import org.junit.runner.RunWith;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "vellamo";
    public static ArrayList<String> scores = new ArrayList();
    public static Boolean wasError = false;

    protected Bundle parameters;
    protected String version;
    protected Boolean browser;
    protected Boolean metal;
    protected Boolean multicore;
    protected Integer browserToUse;
    protected String packageID;

    @Before
    public void initialize(){
        parameters = getParams();
        packageID = getPackageID(parameters);
        version = parameters.getString("version");
        browser = parameters.getBoolean("browser");
        metal = parameters.getBoolean("metal");
        multicore = parameters.getBoolean("multicore");
        browserToUse = parameters.getInt("browserToUse") - 1;
    }

    @Test
    public void setup() throws Exception {
        dismissAndroidVersionPopup();
        dismissEULA();
        if (version.equals("2.0.3")) {
            dissmissWelcomebanner();
        } else {
            dismissLetsRoll();
            if (version.equals("3.2.4")) {
                dismissArrow();
            }
        }
    }

    @Test
    public void runWorkload() throws Exception {
        if (version.equals("2.0.3")) {
            startTest();
            dismissNetworkConnectionDialogIfNecessary();
            dismissExplanationDialogIfNecessary();
            waitForTestCompletion(15 * 60, packageID + "act_ba_results_btn_no");
        } else {
             if (browser) {
                 startBrowserTest(browserToUse, version);
                 proccessTest("Browser");
             }
             if (multicore) {
                 startTestV3(1, version);
                 proccessTest("Multicore");
             }
            if (metal) {
                startTestV3(2, version);
                proccessTest("Metal");
            }
        }
    }

    @Test
    public void extractResults() throws Exception {
        for(String result : scores){
            Log.v(TAG, String.format("VELLAMO RESULT: %s", result));
        }
        if (wasError) Log.v("vellamoWatcher", "VELLAMO ERROR: Something crashed while running browser benchmark");
    }

    public void startTest() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runButton = mDevice.findObject(selector.textContains("Run All Chapters"));

        if (!runButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            UiObject pager = mDevice.findObject(selector.className("android.support.v4.view.ViewPager"));
            pager.swipeLeft(2);
            if (!runButton.exists()) {
                throw new UiObjectNotFoundException("Could not find \"Run All Chapters\" button.");
            }
        }
        runButton.click();
    }

    public void startBrowserTest(int browserToUse, String version) throws Exception {
        //Ensure chrome is selected as "browser" fails to run the benchmark
        UiSelector selector = new UiSelector();
        UiObject browserToUseButton = mDevice.findObject(selector.className("android.widget.ImageButton")
                                               .longClickable(true).instance(browserToUse));
        UiObject browserButton = mDevice.findObject(selector.className("android.widget.ImageButton")
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
                UiObject stoppedWorkingDialog = mDevice.findObject(new UiSelector().textStartsWith("Unfortunately"));
                if(stoppedWorkingDialog.exists()){
                    wasError = true;
                    UiObject okButton = mDevice.findObject(new UiSelector().className("android.widget.Button").text("OK"));
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
        mDevice.registerWatcher("stoppedWorkingDialogWatcher", stoppedWorkingDialogWatcher);

        // Run watcher
        mDevice.runWatchers();

        startTestV3(0, version);
    }

    public void startTestV3(int run, String version) throws Exception {
        UiSelector selector = new UiSelector();

        UiObject thirdRunButton = mDevice.findObject(selector.resourceId(packageID + "card_launcher_run_button").instance(2));
        if (!thirdRunButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!thirdRunButton.exists()) {
                throw new UiObjectNotFoundException("Could not find three \"Run\" buttons.");
            }
        }

        //Run benchmarks
        UiObject runButton = mDevice.findObject(selector.resourceId(packageID + "card_launcher_run_button").instance(run));
        if (!runButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!runButton.exists()) {
                throw new UiObjectNotFoundException("Could not find correct \"Run\" button.");
            }
        }
        runButton.click();

        //Skip tutorial screen
        if (version.equals("3.2.4")) {
            UiObject gotItButton = mDevice.findObject(selector.textContains("Got it"));
            if (!gotItButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
                if (!gotItButton.exists()) {
                    throw new UiObjectNotFoundException("Could not find correct \"GOT IT\" button.");
                }
            }
            gotItButton.click();
        }

        else {
            UiObject swipeScreen = mDevice.findObject(selector.textContains("Swipe left to continue"));
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

    }

    public void proccessTest(String metric) throws Exception{
        waitForTestCompletion(15 * 60, packageID + "button_no");

        //Remove watcher
        mDevice.removeWatcher("stoppedWorkingDialogWatcher");

        getScore(metric, packageID + "card_score_score");
        mDevice.pressBack();
        mDevice.pressBack();
        mDevice.pressBack();
    }

    public void getScore(String metric, String resourceID) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject score = mDevice.findObject(selector.resourceId(resourceID));
        if (!score.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!score.exists()) {
                throw new UiObjectNotFoundException("Could not find score on screen.");
            }
        }
        scores.add(metric + " " + score.getText().trim());
    }

    public void waitForTestCompletion(int timeout, String resourceID) throws Exception {
        UiSelector selector = new UiSelector();
        UiObject resultsNoButton = mDevice.findObject(selector.resourceId(resourceID));
        if (!resultsNoButton.waitForExists(TimeUnit.SECONDS.toMillis(timeout))) {
            throw new UiObjectNotFoundException("Did not see results screen.");
        }

    }

    public void dismissEULA() throws Exception {
        UiSelector selector = new UiSelector();
        waitText("Vellamo EULA");
        UiObject acceptButton = mDevice.findObject(selector.textMatches("Accept|ACCEPT")
                                                     .className("android.widget.Button"));
        if (acceptButton.exists()) {
            acceptButton.click();
        }
    }

    public void dissmissWelcomebanner() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject welcomeBanner = mDevice.findObject(selector.textContains("WELCOME"));
        if (welcomeBanner.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            UiObject pager = mDevice.findObject(selector.className("android.support.v4.view.ViewPager"));
            pager.swipeLeft(2);
            pager.swipeLeft(2);
        }
    }

    public void dismissLetsRoll() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject letsRollButton = mDevice.findObject(selector.className("android.widget.Button")
                                                       .textContains("LET'S ROLL"));
        if (!letsRollButton.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!letsRollButton.exists()) {
            // As a fall-back look for the old capitalization
            letsRollButton = mDevice.findObject(selector.className("android.widget.Button")
                              .textContains("Let's Roll"));
            if (!letsRollButton.exists()) {
            throw new UiObjectNotFoundException("Could not find \"Let's Roll\" button.");
            }
            }
        }
        letsRollButton.click();
    }

    public void dismissArrow() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject cardContainer = mDevice.findObject(selector.resourceId(packageID + "cards_container")) ;
        if (!cardContainer.waitForExists(TimeUnit.SECONDS.toMillis(5))) {
            if (!cardContainer.exists()) {
                throw new UiObjectNotFoundException("Could not find vellamo main screen");
            }
        }
    }

    public void dismissNetworkConnectionDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject dialog = mDevice.findObject(selector.className("android.widget.TextView")
                                               .textContains("No Network Connection"));
        if (dialog.exists()) {
            UiObject yesButton = mDevice.findObject(selector.className("android.widget.Button")
                                                      .text("Yes"));
            yesButton.click();
        }
    }

    public void dismissExplanationDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject dialog = mDevice.findObject(selector.className("android.widget.TextView")
                                               .textContains("Benchmarks Explanation"));
        if (dialog.exists()) {
            UiObject noButton = mDevice.findObject(selector.className("android.widget.Button")
                                                     .text("No"));
            noButton.click();
        }
    }
}
