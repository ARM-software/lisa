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


package com.arm.wlauto.uiauto.quadrant;

import java.util.concurrent.TimeUnit;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {   

    public static String TAG = "quadrant";

    public void runUiAutomation() throws Exception {
        Bundle status = new Bundle();
        Bundle params = getParams();
        boolean hasGpu = Boolean.parseBoolean(params.getString("has_gpu").toLowerCase());

        clearLogcat();
        handleFtuInfoDialogIfNecessary();
        goToRunCustomBenchmark();
        selectTestsToRun(hasGpu);
        hitStart();
        handleWarningIfNecessary();
        waitForResults();

        getAutomationSupport().sendStatus(Activity.RESULT_OK, status);
    }

    public void handleFtuInfoDialogIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject infoText = new UiObject(selector.text("Information"));
        if (infoText.waitForExists(TimeUnit.SECONDS.toMillis(10)))
        {
            UiObject okButton = new UiObject(selector.text("OK")
                                                     .className("android.widget.Button"));
            okButton.click();
        }
        else
        {
            // FTU dialog didn't come up.
        }
    }

    public void goToRunCustomBenchmark() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject runCustom = new UiObject(selector.text("Run custom benchmark")
                                                  .className("android.widget.TextView"));
        runCustom.clickAndWaitForNewWindow();
    }

    // By default, all tests are selected. However, if our device does not have a GPU, then
    // running graphics tests may cause a crash, so we disable those.
    public void selectTestsToRun(boolean hasGpu) throws Exception {
        if(!hasGpu) {
            UiSelector selector = new UiSelector();
            UiObject gfx2d = new UiObject(selector.text("2D graphics")
                                                  .className("android.widget.CheckBox"));
            gfx2d.click();

            UiObject gfx3d = new UiObject(selector.text("3D graphics")
                                                  .className("android.widget.CheckBox"));
            gfx3d.click();
        }       
    }

    public void hitStart() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject startButton = new UiObject(selector.text("Start")
                                                    .className("android.widget.Button")
                                                    .packageName("com.aurorasoftworks.quadrant.ui.professional"));
        startButton.click();
    }

    // Even if graphics tests aren't selected, Quadrant will still show a warning about running
    // with software rendering.
    public void handleWarningIfNecessary() throws Exception {
        UiSelector selector = new UiSelector();
        UiObject  warning = new UiObject(selector.text("Warning"));
        if (warning.waitForExists(TimeUnit.SECONDS.toMillis(2))) {
            UiObject closeButton = new UiObject(selector.text("Close")
                                                        .className("android.widget.Button"));
            if (closeButton.exists()) {
                closeButton.click();
            }
        }
        else
        {
            // Warning dialog didn't come up.
        }
    }

    public void waitForResults() throws Exception {
        waitForLogcatText("benchmark aggregate score is", TimeUnit.SECONDS.toMillis(200));
    }
}
