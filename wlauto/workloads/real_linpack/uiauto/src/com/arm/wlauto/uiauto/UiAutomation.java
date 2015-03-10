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


package com.arm.wlauto.uiauto.reallinpack;

import android.app.Activity;
import android.os.Bundle;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {   

    public void runUiAutomation() throws Exception{
        Bundle status = new Bundle();
        status.putString("product", getUiDevice().getProductName());
        UiSelector selector = new UiSelector(); 
        // set the maximum number of threads
        String maxThreads = getParams().getString("max_threads");
        UiObject maxThreadNumberField = new UiObject(selector.index(3));
        maxThreadNumberField.clearTextField();
        maxThreadNumberField.setText(maxThreads);
        // start the benchamrk
        UiObject btn_st = new UiObject(selector.text("Run"));
        btn_st.click();
        btn_st.waitUntilGone(500);
        // set timeout for the benchmark
        btn_st.waitForExists(60 * 60 * 1000);
        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

}
