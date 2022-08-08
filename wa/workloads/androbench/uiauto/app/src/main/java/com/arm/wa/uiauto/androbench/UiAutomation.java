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


package com.arm.wa.uiauto.androbench;

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
        dismissPermissions();
        dismissAndroidVersionPopup();
    }

    @Test
    public void dismissPermissions() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject cont = mDevice.findObject(selector.textContains("Continue"));

        if (cont.exists()) {
            cont.click();
        }
    }

    @Test
    public void runWorkload() throws Exception {
        runBenchmark();
    }

    @Test
    public void extractResults() throws Exception {
        getScores();
    }

    public void runBenchmark() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject btn_microbench = mDevice.findObject(selector.textContains("Micro")
                                                     .className("android.widget.Button"));
        if (btn_microbench.exists()) {
            btn_microbench.click();
        } else {
            UiObject bench =
                mDevice.findObject(new UiSelector().resourceIdMatches("com.andromeda.androbench2:id/btnStartingBenchmarking"));
            Rect bounds = bench.getBounds();
            mDevice.click(bounds.centerX(), bounds.centerY());
        }
        UiObject btn_yes= mDevice.findObject(selector.textContains("Yes")
                                                     .className("android.widget.Button"));
        btn_yes.click();

        UiObject complete_text = mDevice.findObject(selector.text("Cancel")
                                                        .className("android.widget.Button"));
        waitObject(complete_text);
        sleep(2);
        complete_text.click();
    }

    public void getScores() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject seqRead =
            mDevice.findObject(selector.text("Sequential Read").fromParent(selector.index(1)));
        UiObject seqWrite =
            mDevice.findObject(selector.text("Sequential Write").fromParent(selector.index(1)));
        UiObject ranRead =
            mDevice.findObject(selector.text("Random Read").fromParent(selector.index(1)));
        UiObject ranWrite =
            mDevice.findObject(selector.text("Random Write").fromParent(selector.index(1)));
        UiObject sqlInsert =
            mDevice.findObject(selector.text("SQLite Insert").fromParent(selector.index(1)));
        UiObject sqlUpdate =
            mDevice.findObject(selector.text("SQLite Update").fromParent(selector.index(1)));
        UiObject sqlDelete =
            mDevice.findObject(selector.text("SQLite Delete").fromParent(selector.index(1)));

        UiScrollable scrollView = new UiScrollable(new UiSelector().scrollable(true));
        Log.d(TAG, "Sequential Read Score " + seqRead.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(seqWrite);        }
        Log.d(TAG, "Sequential Write Score " + seqWrite.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(ranRead);}
        Log.d(TAG, "Random Read Score " + ranRead.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(ranWrite);}
        Log.d(TAG, "Random Write Score " + ranWrite.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(sqlInsert);}
        Log.d(TAG, "SQL Insert Score " + sqlInsert.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(sqlUpdate);}
        Log.d(TAG, "SQL Update Score " + sqlUpdate.getText());

        if (scrollView.exists()){scrollView.scrollIntoView(sqlDelete);}
        Log.d(TAG, "SQL Delete Score " + sqlDelete.getText());
    }
}
