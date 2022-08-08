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

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    private int networkTimeoutSecs = 30;
    private long networkTimeout =  TimeUnit.SECONDS.toMillis(networkTimeoutSecs);
    private Boolean isCorporate;
    public static String TAG = "UXPERF";
    protected Bundle parameters;
    protected String[] testList;
    protected String packageID;


    @Before
    public void initialize(){
        parameters = getParams();
        testList = parameters.getStringArray("tests");
        packageID = getPackageID(parameters);
        isCorporate = parameters.getBoolean("is_corporate");
    }

    @Test
    public void setup() throws Exception{
        setScreenOrientation(ScreenOrientation.NATURAL);
        clearFirstRun();

        // Ensure we're on the home screen
        UiObject homeButton = mDevice.findObject(
                new UiSelector().resourceId(packageID + "tabbar_back"))
                                .getChild(new UiSelector().index(0));
        homeButton.click();

        //Calculate the location of the test selection button
        UiObject circle =
            mDevice.findObject(new UiSelector().resourceId(packageID + "main_circleControl")
            .className("android.widget.RelativeLayout"));
        Rect bounds = circle.getBounds();
        int selectx = bounds.width()/4;
        selectx = bounds.centerX() + selectx;
        int selecty = bounds.height()/4;
        selecty = bounds.centerY() + selecty;

        Log.d(TAG, "maxx " + selectx);
        Log.d(TAG, "maxy " + selecty);

        mDevice.click(selectx,selecty);

        // Disable test categories
        toggleTest("High-Level Tests");
        toggleTest("Low-Level Tests");
        toggleTest("Special Tests");
	if (isCorporate)
		toggleTest("Fixed Time Test");

        // Enable selected tests
        for (String test : testList) {
            toggleTest(test);
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
            mDevice.findObject(new UiSelector().resourceId(packageID + "main_view_back")
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
            mDevice.findObject(new UiSelector().resourceId(packageID + "results_testList"));
        complete.waitForExists(1200000);

        UiObject outOfmemory = mDevice.findObject(new UiSelector().text("OUT_OF_MEMORY"));
        if (outOfmemory.exists()) {
            throw new OutOfMemoryError("The workload has failed because the device is doing to much work.");
        }
    }

    public void getScores() throws Exception {
        // To ensure we print all scores, some will be printed multiple times but these are filtered on the python side.
        UiScrollable scrollable = new UiScrollable(new UiSelector().scrollable(true));
        // Start at the bottom of the list as this seems more reliable when extracting results.
        scrollable.flingToEnd(10);
        Boolean top_of_list = false;
        while(true) {
            UiObject resultsList =
                mDevice.findObject(new UiSelector().resourceId(packageID + "results_testList"));
            // Find the element in the list that contains our test and pull result and sub_result
            for (int i=1; i < resultsList.getChildCount(); i++) {
                UiObject testName = resultsList.getChild(new UiSelector().index(i))
                    .getChild(new UiSelector().resourceId(packageID + "updated_result_item_name"));
                UiObject result = resultsList.getChild(new UiSelector()
                                    .index(i))
                                    .getChild(new UiSelector()
                                    .resourceId(packageID + "updated_result_item_result"));
                UiObject subResult = resultsList.getChild(new UiSelector()
                                    .index(i))
                                    .getChild(new UiSelector()
                                    .resourceId(packageID + "updated_result_item_subresult"));
                if (testName.waitForExists(500) && result.waitForExists(500) && subResult.waitForExists(500)) {
                    Log.d(TAG, "name: (" + testName.getText() + ") result: (" + result.getText() + ") sub_result: (" + subResult.getText() + ")");
                }
            }
            // Ensure we loop over the first screen an extra time once the top of the list has been reached.
            if (top_of_list){
                break;
            }
            top_of_list = !scrollable.scrollBackward(100);
        }
    }

    public void toggleTest(String testname) throws Exception {
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true)
                                                .resourceId(packageID + "main_testSelectListView"));
        UiObject test =
            mDevice.findObject(new UiSelector().text(testname));
        if (!test.exists() && list.waitForExists(60)) {
            list.flingToBeginning(10);
            list.scrollIntoView(test);
        }
        test.click();
    }
}
