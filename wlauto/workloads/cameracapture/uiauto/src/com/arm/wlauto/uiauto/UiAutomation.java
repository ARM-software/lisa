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


package com.arm.wlauto.uiauto.cameracapture;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;

import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "cameracapture";

    public void runUiAutomation() throws Exception {
       int timeDurationBetweenEachCapture = 0;
       int sleepTime = 2;
       Bundle parameters = getParams();
       String noOfCaptures = "";
       int iterations = 0;

       if (parameters.size() > 0) {
             iterations = Integer.parseInt(parameters
                          .getString("no_of_captures"));
             timeDurationBetweenEachCapture = Integer.parseInt(parameters
                     .getString("time_between_captures"));
       }
       // switch to camera capture mode
       UiObject clickModes = new UiObject(new UiSelector().descriptionMatches("Camera, video or panorama selector"));
       clickModes.click();
       sleep(sleepTime);

       UiObject changeModeToCapture = new UiObject(new UiSelector().descriptionMatches("Switch to photo"));

       changeModeToCapture.click();
       sleep(sleepTime);

       // click to capture photos
       UiObject clickCaptureButton = new UiObject(new UiSelector().descriptionMatches("Shutter button"));

       for (int i = 0; i < iterations; i++) {
           clickCaptureButton.longClick();
           sleep(timeDurationBetweenEachCapture);
       }
       getUiDevice().pressBack();
    }
}
