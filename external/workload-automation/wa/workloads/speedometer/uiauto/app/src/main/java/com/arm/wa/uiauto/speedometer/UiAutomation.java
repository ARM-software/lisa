/*    Copyright 2014-2018 ARM Limited
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

package com.arm.wa.uiauto.speedometer;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import com.arm.wa.uiauto.BaseUiAutomation;
import android.util.Log;


import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    private int networkTimeoutSecs = 30;
    private long networkTimeout =  TimeUnit.SECONDS.toMillis(networkTimeoutSecs);
    public static String TAG = "UXPERF";
    public boolean textenabled = false;

    @Before
    public void initialize(){
        initialize_instrumentation();
    }

    @Test
    public void setup() throws Exception{
        setScreenOrientation(ScreenOrientation.NATURAL);
        clearFirstRun();
    }

    @Test
    public void runWorkload() throws Exception {
        runBenchmark();
    }

    @Test
    public void teardown() throws Exception{
        clearTabs();
        unsetScreenOrientation();
    }

    public void clearFirstRun() throws Exception {
        UiObject accept =
            mDevice.findObject(new UiSelector().resourceId("com.android.chrome:id/terms_accept")
                .className("android.widget.Button"));
        if (accept.exists()){
            accept.click();
            UiObject negative =
                mDevice.findObject(new UiSelector().resourceId("com.android.chrome:id/negative_button")
                    .className("android.widget.Button"));
            negative.waitForExists(100000);
            negative.click();
        }
    }

    public void runBenchmark() throws Exception {
        UiObject start =
            mDevice.findObject(new UiSelector().description("Start Test")
                .className("android.widget.Button"));
            if (start.waitForExists(2000)){
                start.click();
            } else {
            UiObject starttext =
                mDevice.findObject(new UiSelector().text("Start Test")
                    .className("android.widget.Button"));
                starttext.click();
            }
        UiObject scores =
            mDevice.findObject(new UiSelector().resourceId("result-number")
                .className("android.view.View"));
        scores.waitForExists(2100000);
        String textScore = scores.getText();
        Log.d(TAG, "Speedometer Score " + textScore);
        Log.d(TAG, "Speedometer Score " + scores.getContentDescription());
    }

    public void clearTabs() throws Exception {
        UiObject tabselector =
            mDevice.findObject(new UiSelector().resourceId("com.android.chrome:id/tab_switcher_button")
                .className("android.widget.ImageButton"));
        if (!tabselector.exists()){
            return;
        }
        tabselector.click();
        UiObject menu =
            mDevice.findObject(new UiSelector().resourceId("com.android.chrome:id/menu_button")
                .className("android.widget.ImageButton"));
        menu.click();
        UiObject closetabs =
            mDevice.findObject(new UiSelector().textContains("Close all tabs"));
        if (closetabs.exists()){
            closetabs.click();
        }
    }
}
