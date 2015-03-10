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


package com.arm.wlauto.uiauto.benchmarkpi;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {   

    public static String TAG = "benchmarkpi";

    public void runUiAutomation() throws Exception{
        Bundle status = new Bundle();

	startTest();
	waitForAndExtractResults();

        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

    public void startTest() throws Exception{
        UiSelector selector = new UiSelector();
        UiObject benchButton = new UiObject(selector.text("Benchmark my Android!")
                                                    .className("android.widget.Button"));
        benchButton.click();
    }

    public void waitForAndExtractResults() throws Exception{
        UiSelector selector = new UiSelector();
        UiObject submitButton = new UiObject(selector.text("Submit")
                                                     .className("android.widget.Button"));
	submitButton.waitForExists(10 * 1000);

        UiObject resultsText = new UiObject(selector.textContains("You calculated Pi in")
                                                    .className("android.widget.TextView"));
	Log.v(TAG, resultsText.getText());
    }
}
