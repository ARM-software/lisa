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


package com.arm.wlauto.uiauto.linpack;

import java.util.concurrent.TimeUnit;

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

    public static String TAG = "linpack";

    public void runUiAutomation() throws Exception{
        UiSelector selector = new UiSelector();
        UiObject runSingleButton = new UiObject(selector.text("Run Single Thread"));
        runSingleButton.click();
        runSingleButton.waitUntilGone(500);
        runSingleButton.waitForExists(TimeUnit.SECONDS.toMillis(30));

        UiObject mflops = new UiObject(new UiSelector().className("android.widget.TextView").instance(2));
        Log.v(TAG, String.format("LINPACK RESULT: ST %s", mflops.getText()));

        UiObject runMultiButton = new UiObject(selector.text("Run Multi-Thread"));
        runMultiButton.click();
        runMultiButton.waitUntilGone(500);
        runMultiButton.waitForExists(TimeUnit.SECONDS.toMillis(30));

        Log.v(TAG, String.format("LINPACK RESULT: MT %s", mflops.getText()));

        Bundle status = new Bundle();
        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

}
