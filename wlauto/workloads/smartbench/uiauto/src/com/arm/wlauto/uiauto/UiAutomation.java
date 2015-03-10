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


package com.arm.wlauto.uiauto.smartbench;

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

    public static String TAG = "smartbench";

    public void runUiAutomation() throws Exception {
        Bundle status = new Bundle();
        status.putString("product", getUiDevice().getProductName());
        UiSelector selector = new UiSelector();
        sleep(3);
        UiObject text_bench = new UiObject(selector.text("Run SmartBench")
                                                   .className("android.widget.TextView"));
        text_bench.click();

        try{
            UiObject complete_text = new UiObject(selector .textContains("Display Index Scores")
                                                           .className("android.widget.TextView"));

            waitObject(complete_text);            

            sleep(2);
            complete_text.click();
        } finally{
            //complete_text.click();
        }

        sleep(5);
        takeScreenshot("SmartBench");
        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

}
