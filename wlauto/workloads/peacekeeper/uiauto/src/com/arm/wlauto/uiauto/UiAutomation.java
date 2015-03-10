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


package com.arm.wlauto.uiauto.peacekeeper;

import java.io.File;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.net.URLConnection;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.PrintWriter;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;

// Import the uiautomator libraries
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation { 

    public static String TAG = "peacekeeper";

    public void runUiAutomation() throws Exception {
        // maximum time for running peacekeeper benchmark 80 * 10 sec
        final int TIMEOUT = 80;

        // reading the input parameter
        Bundle parameters = getParams();
        String browser = parameters.getString("browser");
        String outputFile = parameters.getString("output_file");
        String peacekeeperUrl = parameters.getString("peacekeeper_url");

        String urlAddress = "";

        PrintWriter writer = new PrintWriter(outputFile, "UTF-8");

        // firefox browser uiautomator code
        if (browser.equals("firefox")) {

            UiObject addressBar = new UiObject(new UiSelector()
                                  .className("android.widget.TextView")
                                  .text("Enter Search or Address"));
            addressBar.click();
            UiObject setUrl = new UiObject(new UiSelector()
                              .className("android.widget.EditText"));
            setUrl.clearTextField();
            setUrl.setText(peacekeeperUrl);
            getUiDevice().pressEnter();

            UiObject currentUrl = new UiObject(new UiSelector()
                               .className("android.widget.TextView").index(1));
            for (int i = 0; i < TIMEOUT; i++) {

                if (currentUrl.getText()
                   .equals("Peacekeeper - free universal browser test for HTML5 from Futuremark")) {

                    // write url address to peacekeeper.txt file
                    currentUrl.click();
                    urlAddress = setUrl.getText();
                    writer.println(urlAddress);
                    break;
                }
            sleep(10);
            }
        } else if (browser.equals("chrome")) { // Code for Chrome browser
            UiObject adressBar = new UiObject(new UiSelector()
                                  .className("android.widget.EditText")
                                  .description("Search or type url"));

            adressBar.clearTextField();
            adressBar.setText(peacekeeperUrl);
            getUiDevice().pressEnter();
            for (int i = 0; i < TIMEOUT; i++) {

                if (!adressBar.getText().contains("run.action")) {

                    // write url address to peacekeeper.txt file
                    urlAddress = adressBar.getText();
                    if (!urlAddress.contains("http"))
                    urlAddress = "http://" + urlAddress;
                    writer.println(urlAddress);
                    break;
                    }
            sleep(10);
            }
        }
        writer.close();
        getUiDevice().pressHome();
    }
}
