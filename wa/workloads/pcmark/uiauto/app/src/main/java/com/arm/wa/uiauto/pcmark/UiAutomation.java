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

package com.arm.wa.uiauto.pcmark;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ActionLogger;
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
    public static final long WAIT_TIMEOUT_5SEC = TimeUnit.SECONDS.toMillis(5);

    @Before
    public void initialize(){
        initialize_instrumentation();
    }

    @Test
    public void setup() throws Exception{
        dismissAndroidVersionPopup();
        clearPopups();
        setScreenOrientation(ScreenOrientation.PORTRAIT);
        loadBenchmarks();
        installBenchmark();
    }

    @Test
    public void runWorkload() throws Exception {
        runBenchmark();
    }

    @Test
    public void teardown() throws Exception{
        unsetScreenOrientation();
    }

    private void clearPopups() throws Exception{
        UiObject permiss =
            mDevice.findObject(new UiSelector().textMatches("(?i)Continue"));
        if (permiss.exists()){
            permiss.click();
        }
        UiObject compat =
            mDevice.findObject(new UiSelector().text("OK"));
        if (compat.exists()){
            compat.click();
            if (compat.exists()){
                compat.click();
            }
        }
    }

    //Swipe to benchmarks and back to initialise the app correctly
    private void loadBenchmarks() throws Exception {
        UiObject title =
            mDevice.findObject(new UiSelector().text("PCMARK"));
        title.waitForExists(300000);
        if (title.exists()){
            title.click();
            UiObject benchPage = getUiObjectByText("BENCHMARKS");
            benchPage.waitForExists(60000);
            benchPage.click();
            benchPage.click();
            UiObject pcmark = getUiObjectByText("PCMARK");
            pcmark.waitForExists(60000);
            pcmark.click();
        } else {
            throw new UiObjectNotFoundException("Application has not loaded within the given time");
        }
    }

    //Install the Work 2.0 Performance Benchmark
    private void installBenchmark() throws Exception {
        UiObject benchmark =
            mDevice.findObject(new UiSelector().descriptionContains("INSTALL("));
        if (benchmark.exists()) {
            benchmark.click();
        } else {
            UiObject benchmarktext =
                mDevice.findObject(new UiSelector().textContains("INSTALL("));
            if(benchmarktext.exists()) {
                benchmarktext.click();
            }
        }
        
        UiObject install =
            mDevice.findObject(new UiSelector().description("INSTALL")
                .className("android.view.View"));
        if (install.exists()) {
            install.click();
        } else {
            UiObject installtext =
                mDevice.findObject(new UiSelector().text("INSTALL")
                       .className("android.view.View"));
            if (installtext.exists()) {
                installtext.click();
            }
        }
        UiObject installed =
            mDevice.findObject(new UiSelector().description("RUN")
                    .className("android.view.View"));
            installed.waitForExists(360000);
            if (!installed.exists()){
                UiObject installedtext =
                    mDevice.findObject(new UiSelector().text("RUN")
                           .className("android.view.View"));
                    installedtext.waitForExists(1000);
            }
    }

    //Execute the Work 2.0 Performance Benchmark - wait up to ten minutes for this to complete
    private void runBenchmark() throws Exception {
    	// After installing, stop screen switching back to landscape. 
    	setScreenOrientation(ScreenOrientation.PORTRAIT);
        UiObject run =
            mDevice.findObject(new UiSelector().resourceId("CONTROL_PCMA_WORK_V2_DEFAULT")
                                               .className("android.view.View")
                                               .childSelector(new UiSelector().text("RUN")
                                               .className("android.view.View")));
        if (run.exists()) {
            run.clickTopLeft();
        } else {
            UiObject runtext =
                mDevice.findObject(new UiSelector().text("RUN"));
                if (runtext.waitForExists(2000)) {
                    runtext.click();
                } else {
                    UiObject rundesc =
                        mDevice.findObject(new UiSelector().description("RUN"));
                    rundesc.click();
                }
        }
        UiObject score =
            mDevice.findObject(new UiSelector().text("SCORE DETAILS")
                .className("android.widget.TextView"));
        if (!score.waitForExists(3600000)){
            throw new UiObjectNotFoundException("Workload has not completed within the given time");
        }
    }
}
