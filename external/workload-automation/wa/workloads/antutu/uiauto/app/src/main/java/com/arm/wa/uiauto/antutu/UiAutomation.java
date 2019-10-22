/*    Copyright 2013-2018 ARM Limited
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


package com.arm.wa.uiauto.antutu;

import android.app.Activity;
import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiScrollable;
import android.support.test.uiautomator.UiSelector;
import android.util.Log;

import com.arm.wa.uiauto.BaseUiAutomation;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "UXPERF";
    public static String TestButton5 = "com.antutu.ABenchMark:id/start_test_region";
    public static String TestButton6 = "com.antutu.ABenchMark:id/start_test_text";
    private static int initialTimeoutSeconds = 20;

    @Test
    public void setup() throws Exception {
       dismissAndroidVersionPopup();
       clearPopups();
    }

    @Test
    public void runWorkload() throws Exception{
        hitTest();
        waitforCompletion();
    }

    @Test
    public void extractResults() throws Exception{
        getScores();
    }

    public void hitTest() throws Exception {
        UiObject testbutton =
            mDevice.findObject(new UiSelector().resourceId("com.antutu.ABenchMark:id/main_test_start_title"));
        testbutton.click();
        sleep(1);
    }

    public void clearPopups() throws Exception {
        UiObject cancel =
            mDevice.findObject(new UiSelector().textContains("CANCEL"));
        cancel.waitForExists(5000);
        if (cancel.exists()){
            cancel.click();
        }
    }

    public void waitforCompletion() throws Exception {
        UiObject totalScore =
            mDevice.findObject(new UiSelector().resourceId("com.antutu.ABenchMark:id/textViewTotalScore"));
        totalScore.waitForExists(600000);
    }

    public void getScores() throws Exception {
        //Expand, Extract and Close CPU sub scores
        UiObject cpuscores =
            mDevice.findObject(new UiSelector().text("CPU"));
        cpuscores.click();
        UiObject cpumaths =
            mDevice.findObject(new UiSelector().text("CPU Mathematics Score").fromParent(new UiSelector().index(3)));
        UiObject cpucommon =
            mDevice.findObject(new UiSelector().text("CPU Common Use Score").fromParent(new UiSelector().index(3)));
        UiObject cpumulti =
            mDevice.findObject(new UiSelector().text("CPU Multi-Core Score").fromParent(new UiSelector().index(3)));
        Log.d(TAG, "CPU Maths Score " + cpumaths.getText());
        Log.d(TAG, "CPU Common Score " + cpucommon.getText());
        Log.d(TAG, "CPU Multi Score " + cpumulti.getText());
        cpuscores.click();

        //Expand, Extract and Close GPU sub scores
        UiObject gpuscores =
            mDevice.findObject(new UiSelector().text("GPU"));
        gpuscores.click();
        UiObject gpumaroon =
            mDevice.findObject(new UiSelector().text("3D [Marooned] Score").fromParent(new UiSelector().index(3)));
        UiObject gpucoast =
            mDevice.findObject(new UiSelector().text("3D [Coastline] Score").fromParent(new UiSelector().index(3)));
        UiObject gpurefinery =
            mDevice.findObject(new UiSelector().text("3D [Refinery] Score").fromParent(new UiSelector().index(3)));
        Log.d(TAG, "GPU Marooned Score " + gpumaroon.getText());
        Log.d(TAG, "GPU Coastline Score " + gpucoast.getText());
        Log.d(TAG, "GPU Refinery Score " + gpurefinery.getText());
        gpuscores.click();

        //Expand, Extract and Close UX sub scores
        UiObject uxscores =
            mDevice.findObject(new UiSelector().text("UX"));
        uxscores.click();
        UiObject security =
            mDevice.findObject(new UiSelector().text("Data Security Score").fromParent(new UiSelector().index(3)));
        UiObject dataprocessing =
            mDevice.findObject(new UiSelector().text("Data Processing Score").fromParent(new UiSelector().index(3)));
        UiObject imageprocessing =
            mDevice.findObject(new UiSelector().text("Image Processing Score").fromParent(new UiSelector().index(3)));
        UiObject uxscore =
            mDevice.findObject(new UiSelector().text("User Experience Score").fromParent(new UiSelector().index(3)));
        Log.d(TAG, "Data Security Score " + security.getText());
        Log.d(TAG, "Data Processing Score " + dataprocessing.getText());
        Log.d(TAG, "Image Processing Score " + imageprocessing.getText());
        Log.d(TAG, "User Experience Score " + uxscore.getText());
        uxscores.click();

        //Expand, Extract and Close MEM sub scores
        UiObject memscores =
            mDevice.findObject(new UiSelector().text("MEM"));
        memscores.click();
        UiObject ramscore =
            mDevice.findObject(new UiSelector().text("RAM Score").fromParent(new UiSelector().index(3)));
        UiObject romscore =
            mDevice.findObject(new UiSelector().text("ROM Score").fromParent(new UiSelector().index(3)));
        Log.d(TAG, "RAM Score " + ramscore.getText());
        Log.d(TAG, "ROM Score " + romscore.getText());
        memscores.click();
    }

}
