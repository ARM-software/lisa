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


package com.arm.wlauto.uiauto.sqlite;

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

    public static String TAG = "sqlite";

    public void runUiAutomation() throws Exception {
        Bundle status = new Bundle();
        status.putString("product", getUiDevice().getProductName());
        UiSelector selector = new UiSelector();

        UiObject text_start = new UiObject(selector.text("Start")
                                                   .className("android.widget.Button"));
        text_start.click();

        try {
            UiObject stop_text = new UiObject(selector.textContains("Stop")
                                                      .className("android.widget.Button"));
            waitUntilNoObject(stop_text, 600);

            sleep(2);
            this.extractResults();
        } finally {
        }
    }

    public void extractResults() throws UiObjectNotFoundException{
        UiSelector selector = new UiSelector();
        UiScrollable resultList = new UiScrollable(selector.className("android.widget.ScrollView"));
        resultList.scrollToBeginning(5);
        selector = resultList.getSelector();
        int index = 0;
        while (true){
            UiObject lastEntry = new UiObject(selector.childSelector(new UiSelector()
                                                                    .className("android.widget.LinearLayout")
                                                                    .childSelector(new UiSelector()
                                                                    .index(index)
                                                                    .childSelector(new UiSelector()
                                                                    .className("android.widget.LinearLayout")))));
            if (lastEntry.exists()){
                UiObject value = new UiObject(selector.childSelector(new UiSelector()
                                                                    .className("android.widget.LinearLayout")
                                                                    .childSelector(new UiSelector()
                                                                    .index(index)
                                                                    .childSelector(new UiSelector()
                                                                    .resourceIdMatches(".*test_result.*")))));
                Log.v("sqlite", "Overall = " + value.getText().replace("\n", " "));
                break;
            }

            UiObject label = new UiObject(selector.childSelector(new UiSelector()
                                                                .className("android.widget.LinearLayout")
                                                                .childSelector(new UiSelector()
                                                                .index(index)
                                                                .childSelector(new UiSelector()
                                                                .index(0)))));
            UiObject value = new UiObject(selector.childSelector(new UiSelector()
                                                                .className("android.widget.LinearLayout")
                                                                .childSelector(new UiSelector()
                                                                .index(index)
                                                                .childSelector(new UiSelector()
                                                                .index(1)))));
            index++;
            if (!label.exists()){
                resultList.scrollForward();
                index--;
                sleep(1);
                continue;
            }
            Log.v("sqlite", label.getText() + " = " + value.getText().replace("\n", " "));
        }
    }
}
