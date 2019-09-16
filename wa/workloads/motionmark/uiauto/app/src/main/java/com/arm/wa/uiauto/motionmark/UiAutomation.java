/*    Copyright 2014-2019 ARM Limited
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

package com.arm.wa.uiauto.motionmark;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiScrollable;

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
        setScreenOrientation(ScreenOrientation.LANDSCAPE);
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true));

        UiObject start =
            mDevice.findObject(new UiSelector().text("Run Benchmark")
                .className("android.widget.Button"));
        list.swipeUp(10);
        if (start.exists()){
            start.click();
        } else {
            UiObject startDesc =
                mDevice.findObject(new UiSelector().description("Run Benchmark")
                    .className("android.widget.Button"));
            startDesc.click();
        }

        UiObject results =
            mDevice.findObject(new UiSelector().resourceId("results-score")
                .className("android.widget.GridView"));
        results.waitForExists(2100000);

        setScreenOrientation(ScreenOrientation.PORTRAIT);

        UiObject multiply = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(2))
            .getChild(new UiSelector().index(0));            
        Log.d(TAG, "Multiply Score " + multiply.getText());

        UiObject canvas = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(3))
            .getChild(new UiSelector().index(0));
        Log.d(TAG, "Canvas Score " + canvas.getText());

        UiObject leaves = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(4))
            .getChild(new UiSelector().index(0));
        Log.d(TAG, "Leaves Score " + leaves.getText());

        UiObject paths = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(5))
            .getChild(new UiSelector().index(0));
        Log.d(TAG, "Paths Score " + paths.getText());

        UiObject canvaslines = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(6))
            .getChild(new UiSelector().index(0));
        if (!canvaslines.exists() && list.waitForExists(60)) {
            list.scrollIntoView(canvaslines);
        }
        Log.d(TAG, "Canvas Lines Score " + canvaslines.getText());

        UiObject focus = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(7))
            .getChild(new UiSelector().index(0));
        if (!focus.exists() && list.waitForExists(60)) {
            list.scrollIntoView(focus);
        }
        Log.d(TAG, "Focus Score " + focus.getText());

        UiObject images = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(8))
            .getChild(new UiSelector().index(0));
        if (!images.exists() && list.waitForExists(60)) {
            list.scrollIntoView(images);
        }
        Log.d(TAG, "Images Score " + images.getText());

        UiObject design = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(9))
            .getChild(new UiSelector().index(0));
        if (!design.exists() && list.waitForExists(60)) {
            list.scrollIntoView(design);
        }
        Log.d(TAG, "Design Score " + design.getText());

        UiObject suits = 
            mDevice.findObject(new UiSelector().resourceId("results-score"))
            .getChild(new UiSelector().index(10))
            .getChild(new UiSelector().index(0));
        if (!suits.exists() && list.waitForExists(60)) {
            list.scrollIntoView(suits);
        }
        Log.d(TAG, "Suits Score " + suits.getText());
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
        closetabs.click();
    }
}
