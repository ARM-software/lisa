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


package com.arm.wa.uiauto.aitutu;

import android.app.Activity;
import android.os.Bundle;
import android.graphics.Rect;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiScrollable;
import android.view.KeyEvent;
import android.util.Log;

import com.arm.wa.uiauto.BaseUiAutomation;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.TimeUnit;


@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "UXPERF";

    @Test
    public void setup() throws Exception {
        clearPopups();
        downloadAssets();
    }

    @Test
    public void runWorkload() throws Exception {
        runBenchmark();
    }

    @Test
    public void extractResults() throws Exception {
        getScores();
    }

    public void clearPopups() throws Exception {

        UiObject agreement =
            mDevice.findObject(new UiSelector().textContains("NEXT"));
        agreement.waitForExists(5000);
        if (agreement.exists()) {
            agreement.click();
        }

        UiSelector selector = new UiSelector();

        UiObject cancel = mDevice.findObject(selector.textContains("CANCEL")
                                             .className("android.widget.Button"));
        cancel.waitForExists(60000);
        if (cancel.exists()){
            cancel.click();
        }
    }

    public void downloadAssets() throws Exception {
        UiSelector selector = new UiSelector();
        //Start the tests
        UiObject start = mDevice.findObject(selector.textContains("Start Testing")
                                                     .className("android.widget.TextView"));
        waitObject(start);
        start.click();

        UiObject check = mDevice.findObject(selector.textContains("classification")
                                                     .className("android.widget.TextView"));
        waitObject(check);
    }

    public void runBenchmark() throws Exception {
        UiSelector selector = new UiSelector();

        //Wait for the tests to complete
        UiObject complete =
            mDevice.findObject(selector.text("TEST AGAIN")
                .className("android.widget.Button"));
        complete.waitForExists(1200000);

    }

    public void getScores() throws Exception {
        mDevice.waitForIdle(5000);
        UiSelector selector = new UiSelector();
        //Declare the models used
        UiObject imageMod =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(1))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewAIModelName"));
        UiObject objectMod =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(4))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewAIModelName"));
        //Log the scores and models
        UiObject totalScore =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/textViewTotalScore"));
        Log.d(TAG, "Overall Score " + totalScore.getText());
        UiObject imageTotal =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(1))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewSIDScore"));
        Log.d(TAG, "Image Total Score " + imageTotal.getText() + " Model " + imageMod.getText());
        UiObject imageSpeed =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(2))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewBIDScore"));
        Log.d(TAG, "Image Speed Score " + imageSpeed.getText() + " Model " + imageMod.getText());
        UiObject imageAcc =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(3))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewBIDScore"));
        Log.d(TAG, "Image Accuracy Score " + imageAcc.getText() + " Model " + imageMod.getText());
        UiObject objectTotal =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(4))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewSIDScore"));
        Log.d(TAG, "Object Total Score " + objectTotal.getText() + " Model " + objectMod.getText());
        UiObject objectSpeed =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(5))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewBIDScore"));
        Log.d(TAG, "Object Speed Score " + objectSpeed.getText() + " Model " + objectMod.getText());
        UiObject objectAcc =
            mDevice.findObject(selector.resourceId("com.antutu.aibenchmark:id/recyclerView"))
            .getChild(selector.index(6))
            .getChild(selector.resourceId("com.antutu.aibenchmark:id/textViewBIDScore"));
        Log.d(TAG, "Object Accuracy Score " + objectAcc.getText() + " Model " + objectMod.getText());
    }
}
