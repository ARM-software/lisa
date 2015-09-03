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


package com.arm.wlauto.uiauto.andebench;

import java.util.concurrent.TimeUnit;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "andebench";

    private static int initialTimeoutSeconds = 20;
    private static int shortDelaySeconds = 3;

    public void runUiAutomation() throws Exception{
        Bundle status = new Bundle();
        Bundle params = getParams();
        String numThreads = params.getString("number_of_threads");
        Boolean nativeOnly = Boolean.parseBoolean(params.getString("native_only"));
        status.putString("product", getUiDevice().getProductName());

        waitForStartButton();
        setConfiguration(numThreads, nativeOnly);
        hitStart();
        waitForAndExtractResuts();

        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

    public void waitForStartButton() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject startButton = new UiObject(selector.className("android.widget.ImageButton")
                                                    .packageName("com.eembc.coremark"));
        if (!startButton.waitForExists(TimeUnit.SECONDS.toMillis(initialTimeoutSeconds))) {
                throw new UiObjectNotFoundException("Did not see start button.");
        }
    }

    public void setConfiguration(String numThreads, boolean nativeOnly) throws Exception {
        UiSelector selector = new UiSelector();
        getUiDevice().pressMenu();

        UiObject settingsButton = new UiObject(selector.clickable(true));
        settingsButton.click();

        if (nativeOnly) {
            UiObject nativeButton = new UiObject(selector.textContains("Native"));
            nativeButton.click();
        }

        UiObject threadNumberField = new UiObject(selector.className("android.widget.EditText"));
        threadNumberField.clearTextField();
        threadNumberField.setText(numThreads);

        getUiDevice().pressBack();
        sleep(shortDelaySeconds);
        // If the device does not have a physical keyboard, a virtual one might have
        // poped up when setting the number of threads. If that happend, then the above
        // backpress would dismiss the vkb and another one will be necessary to return
        // from the settings screen.
        if(threadNumberField.exists())
        {
            getUiDevice().pressBack();
            sleep(shortDelaySeconds);
        }
    }

    public void hitStart() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject startButton = new UiObject(selector.className("android.widget.ImageButton")
                                                    .packageName("com.eembc.coremark"));
        startButton.click();
        sleep(shortDelaySeconds);
    }

    public void waitForAndExtractResuts() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runningText = new UiObject(selector.textContains("Running...")
                                                    .className("android.widget.TextView")
                                                    .packageName("com.eembc.coremark"));
        runningText.waitUntilGone(TimeUnit.SECONDS.toMillis(600));

        UiObject resultText = new UiObject(selector.textContains("Results in Iterations/sec:")
                                                    .className("android.widget.TextView")
                                                    .packageName("com.eembc.coremark"));
        resultText.waitForExists(TimeUnit.SECONDS.toMillis(shortDelaySeconds));
        Log.v(TAG, resultText.getText());
        sleep(shortDelaySeconds);
    }
}
