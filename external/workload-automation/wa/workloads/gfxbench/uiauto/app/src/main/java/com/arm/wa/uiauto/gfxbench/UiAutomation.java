/*    Copyright 2014-2016 ARM Limited
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

package com.arm.wa.uiauto.gfxbench;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiScrollable;
import android.util.Log;
import android.graphics.Rect;

import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ActionLogger;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    private int networkTimeoutSecs = 30;
    private long networkTimeout =  TimeUnit.SECONDS.toMillis(networkTimeoutSecs);
    public static String TAG = "UXPERF";

    @Before
    public void initialize(){
        initialize_instrumentation();
    }

    @Test
    public void setup() throws Exception{
        setScreenOrientation(ScreenOrientation.NATURAL);
        clearFirstRun();

        //Calculate the location of the test selection button
        UiObject circle =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/main_circleControl")
            .className("android.widget.RelativeLayout"));
        Rect bounds = circle.getBounds();
        int selectx = bounds.width()/4;
        selectx = bounds.centerX() + selectx;
        int selecty = bounds.height()/4;
        selecty = bounds.centerY() + selecty;

        Log.d(TAG, "maxx " + selectx);
        Log.d(TAG, "maxy " + selecty);

        mDevice.click(selectx,selecty);

        //Disable the tests
        toggleTest("High-Level Tests");
        toggleTest("Low-Level Tests");
        toggleTest("Special Tests");
        toggleTest("Fixed Time Test");

        //Enable sub tests
        toggleTest("Car Chase");
        toggleTest("1080p Car Chase Offscreen");
        toggleTest("Manhattan 3.1");
        toggleTest("1080p Manhattan 3.1 Offscreen");
        toggleTest("1440p Manhattan 3.1.1 Offscreen");
        toggleTest("Tessellation");
        toggleTest("1080p Tessellation Offscreen");
    }

    @Test
    public void runWorkload() throws Exception {
        runBenchmark();
        getScores();
    }

    @Test
    public void teardown() throws Exception{
        unsetScreenOrientation();
    }

    public void clearFirstRun() throws Exception {
        UiObject accept =
            mDevice.findObject(new UiSelector().resourceId("android:id/button1")
                .className("android.widget.Button"));
        if (accept.exists()){
            accept.click();
            sleep(5);
        }
        UiObject sync =
                mDevice.findObject(new UiSelector().text("Data synchronization")
                    .className("android.widget.TextView"));
        if (!sync.exists()){
            sync = mDevice.findObject(new UiSelector().text("Pushed data not found")
                    .className("android.widget.TextView"));
        }
        if (sync.exists()){
            UiObject data =
                mDevice.findObject(new UiSelector().resourceId("android:id/button1")
                    .className("android.widget.Button"));
            data.click();
        }

        UiObject home =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/main_homeBack")
                .className("android.widget.LinearLayout"));
            home.waitForExists(300000);
    }

    public void runBenchmark() throws Exception {
        //Start the tests
        UiObject start =
            mDevice.findObject(new UiSelector().text("Start"));
        start.click();

        //Wait for results
        UiObject complete =
            mDevice.findObject(new UiSelector().text("High-Level Tests")
                .className("android.widget.TextView"));
        complete.waitForExists(1200000);
    }

    public void getScores() throws Exception {
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true));
        UiObject results =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"));
        int number_of_results = results.getChildCount();

        //High Level Tests
        UiObject carchase =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(1))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        Log.d(TAG, "Car Chase score " + carchase.getText());

        UiObject carchaseoff =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(2))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        Log.d(TAG, "Car Chase Offscreen score " + carchaseoff.getText());

        UiObject manhattan =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(3))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        Log.d(TAG, "Manhattan 3.1 score " + manhattan.getText());

        UiObject manhattan1080 =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(4))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        Log.d(TAG, "1080p Manhattan 3.1 Offscreen score " + manhattan1080.getText());

        UiObject manhattan1440 =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(5))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        Log.d(TAG, "1440p Manhattan 3.1 Offscreen score " + manhattan1440.getText());

        //Low Level Tests
        UiObject tess =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(7))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        if (!tess.exists() && list.waitForExists(60)) {
            list.scrollIntoView(tess);
        }
        Log.d(TAG, "Tessellation score " + tess.getText());

        UiObject tessoff =
            mDevice.findObject(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/results_testList"))
            .getChild(new UiSelector().index(8))
            .getChild(new UiSelector().resourceId("net.kishonti.gfxbench.gl.v50000.corporate:id/updated_result_item_subresult"));
        if (!tessoff.exists() && list.waitForExists(60)) {
            list.scrollIntoView(tessoff);
        }
        Log.d(TAG, "Tessellation Offscreen score " + tessoff.getText());
    }

    public void toggleTest(String testname) throws Exception {
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true));
        UiObject test =
            mDevice.findObject(new UiSelector().text(testname));
        if (!test.exists() && list.waitForExists(60)) {
            list.scrollIntoView(test);
        }
        test.click();
    }
}
