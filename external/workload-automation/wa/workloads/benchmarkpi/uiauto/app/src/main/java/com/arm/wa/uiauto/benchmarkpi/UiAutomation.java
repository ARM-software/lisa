/*    Copyright 2013-2017 ARM Limited
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


package com.arm.wa.uiauto.benchmarkpi;

import android.app.Activity;
import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiSelector;
import android.util.Log;

import com.arm.wa.uiauto.BaseUiAutomation;

import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "benchmarkpi";

    public Bundle parameters;
    public String packageID;

    @Test
    public void setup() throws Exception {
        dismissAndroidVersionPopup();
    }

    @Test
    public void runWorkload() throws Exception {
        startTest();
        waitForResults();
    }

    @Test
    public void extractResults() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject resultsText = mDevice.findObject(selector.textContains("You calculated Pi in")
                                                          .className("android.widget.TextView"));
        Log.v(TAG, resultsText.getText());
    }

    public void startTest() throws Exception{
        UiSelector selector = new UiSelector();
        UiObject benchButton = mDevice.findObject(selector.text("Benchmark my Android!")
                                                          .className("android.widget.Button"));
        benchButton.click();
    }

    public void waitForResults() throws Exception{
        UiSelector selector = new UiSelector();
        UiObject submitButton = mDevice.findObject(selector.text("Submit")
                                                           .className("android.widget.Button"));
        submitButton.waitForExists(10 * 1000);
    }
}
