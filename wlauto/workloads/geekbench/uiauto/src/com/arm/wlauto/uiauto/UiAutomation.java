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


package com.arm.wlauto.uiauto.geekbench;

import java.util.concurrent.TimeUnit;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {   

    public static String TAG = "geekbench";

    public void runUiAutomation() throws Exception {
        Bundle params = getParams();
        int version = Integer.parseInt(params.getString("version"));
        int times = Integer.parseInt(params.getString("times"));

        for (int i = 0; i < times; i++) {
                runBenchmarks();
                switch(version) {
                case 2: 
                        // In version 2, we scroll through the results WebView to make sure
                        // all results appear on the screen, which causes them to be dumped into
                        // logcat by the Linaro hacks.
                        waitForResultsv2();
                        scrollThroughResults();
                        break;
                case 3: 
                        // Attempting to share the results will generate the .gb3 file with
                        // results that can then be pulled from the device. This is not possible
                        // in verison 2 of Geekbench (Share option was added later).
                        waitForResultsv3();
                        shareResults();
                        break;
                }

                if (i < (times - 1)) {
                        getUiDevice().pressBack();
                        getUiDevice().pressBack();  // twice
                }
        }

        Bundle status = new Bundle();
        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

    public void runBenchmarks() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runButton = new UiObject(selector.text("Run Benchmarks")
                                                  .className("android.widget.Button"));
        if (!runButton.exists()) {
            getUiDevice().pressBack();
        }
        runButton.click();
    }

    public void waitForResultsv2() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject resultsWebview = new UiObject(selector.className("android.webkit.WebView"));
        if (!resultsWebview.waitForExists(TimeUnit.SECONDS.toMillis(200))) {
                throw new UiObjectNotFoundException("Did not see Geekbench results screen.");
        }
    }

    public void waitForResultsv3() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runningTextView = new UiObject(selector.text("Running Benchmarks...")
                                                        .className("android.widget.TextView"));
        runningTextView.waitForExists(TimeUnit.SECONDS.toMillis(2));
        if (!runningTextView.waitUntilGone(TimeUnit.SECONDS.toMillis(200))) {
                throw new UiObjectNotFoundException("Did not get to Geekbench results screen.");
        }
    }

    public void scrollThroughResults() throws Exception {
        UiSelector selector = new UiSelector();
        getUiDevice().pressKeyCode(KeyEvent.KEYCODE_PAGE_DOWN);
        sleep(1);
        getUiDevice().pressKeyCode(KeyEvent.KEYCODE_PAGE_DOWN);
        sleep(1);
        getUiDevice().pressKeyCode(KeyEvent.KEYCODE_PAGE_DOWN);
        sleep(1);
        getUiDevice().pressKeyCode(KeyEvent.KEYCODE_PAGE_DOWN);
    }

    public void shareResults() throws Exception {
	sleep(2); // transition
        UiSelector selector = new UiSelector();
        getUiDevice().pressMenu();
        UiObject runButton = new UiObject(selector.text("Share")
                                                  .className("android.widget.TextView"));
        runButton.waitForExists(500);
        runButton.click();
    }
}
