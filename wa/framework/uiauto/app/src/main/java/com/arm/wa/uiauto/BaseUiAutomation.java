/*    Copyright 2013-2016 ARM Limited
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

package com.arm.wa.uiauto;

import android.app.Instrumentation;
import android.content.Context;
import android.os.SystemClock;
import android.support.test.InstrumentationRegistry;
import android.support.test.uiautomator.UiDevice;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import org.junit.Before;
import org.junit.Test;
import static android.support.test.InstrumentationRegistry.getArguments;

import android.os.Bundle;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.concurrent.TimeoutException;


public class BaseUiAutomation {

    public Instrumentation mInstrumentation;
    public Context mContext;
    public UiDevice mDevice;

    @Before
    public void initialize_instrumentation() {
        mInstrumentation = InstrumentationRegistry.getInstrumentation();
        mDevice = UiDevice.getInstance(mInstrumentation);
        mContext = mInstrumentation.getTargetContext();
    }

    @Test
    public void setup() throws Exception {
    }

    @Test
    public void runWorkload() throws Exception {
    }

    @Test
    public void extractResults() throws Exception {
    }

    @Test
    public void teardown() throws Exception {
    }

    public void sleep(int second) {
        SystemClock.sleep(second * 1000);
    }

    public boolean takeScreenshot(String name) {
        Bundle params = getArguments();
        String png_dir = params.getString("workdir");

        try {
            return mDevice.takeScreenshot(new File(png_dir, name + ".png"));
        } catch (NoSuchMethodError e) {
            return true;
        }
    }

    public void waitText(String text) throws UiObjectNotFoundException {
        waitText(text, 600);
    }

    public void waitText(String text, int second) throws UiObjectNotFoundException {
        UiSelector selector = new UiSelector();
        UiObject text_obj = mDevice.findObject(selector.text(text)
                                                       .className("android.widget.TextView"));
        waitObject(text_obj, second);
    }

    public void waitObject(UiObject obj) throws UiObjectNotFoundException {
        waitObject(obj, 600);
    }

    public void waitObject(UiObject obj, int second) throws UiObjectNotFoundException {
        if (!obj.waitForExists(second * 1000)) {
            throw new UiObjectNotFoundException("UiObject is not found: "
                    + obj.getSelector().toString());
        }
    }

    public boolean waitUntilNoObject(UiObject obj, int second) {
        return obj.waitUntilGone(second * 1000);
    }

    public void clearLogcat() throws Exception {
        Runtime.getRuntime().exec("logcat -c");
    }

    public void waitForLogcatText(String searchText, long timeout) throws Exception {
        long startTime = System.currentTimeMillis();
        Process process = Runtime.getRuntime().exec("logcat");
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;

        long currentTime = System.currentTimeMillis();
        boolean found = false;
        while ((currentTime - startTime) < timeout) {
            sleep(2);  // poll every two seconds

            while ((line = reader.readLine()) != null) {
                if (line.contains(searchText)) {
                    found = true;
                    break;
                }
            }

            if (found) {
                break;
            }
            currentTime = System.currentTimeMillis();
        }

        process.destroy();

        if ((currentTime - startTime) >= timeout) {
            throw new TimeoutException("Timed out waiting for Logcat text \"%s\"".format(searchText));
        }
    }
}
