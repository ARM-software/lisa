/*    Copyright 2018 ARM Limited
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

package com.arm.wa.uiauto.chrome;

import android.app.Activity;
import android.os.Bundle;
import org.junit.Test;
import org.junit.runner.RunWith;
import android.support.test.runner.AndroidJUnit4;

import android.util.Log;
import android.view.KeyEvent;

import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiScrollable;
import android.support.test.uiautomator.UiSelector;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.UiAutoUtils;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation implements ApplaunchInterface {

    protected Bundle parameters;
    protected String packageID;

    public static String TAG = "chrome";

    @Before
    public void initialize() throws Exception {
        parameters = getParams();
        packageID = getPackageID(parameters);
    }

    @Test
    public void setup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        runApplicationSetup();
    }

    public void navigateToPage(String url, boolean from_new_tab) throws Exception {
        UiObject searchBar, urlBar;

        if (from_new_tab) {
                // On the new tab page, click on the search box to turn it into a url bar
                searchBar = mDevice.findObject(new UiSelector().resourceId(packageID + "search_box_text")
                                                               .className("android.widget.EditText"));
                searchBar.click();
        }

        // Navigate to the specified URL
        urlBar = mDevice.findObject(new UiSelector().resourceId(packageID + "url_bar")
                                                    .className("android.widget.EditText"));
        urlBar.click();
        urlBar.setText(url);
        pressEnter();
    }

    public void newTab() throws Exception {
        UiObject tabSwitcher, newTab;

        // Activate the tab switcher
        tabSwitcher = mDevice.findObject(new UiSelector().resourceId(packageID + "tab_switcher_button")
                                                         .className("android.widget.ImageButton"));
        if (tabSwitcher.exists()){
            tabSwitcher.clickAndWaitForNewWindow(uiAutoTimeout);
            // Click the New Tab button
            newTab = mDevice.findObject(new UiSelector().resourceId(packageID + "new_tab_button")
                                                        .className("android.widget.Button"));
            newTab.clickAndWaitForNewWindow(uiAutoTimeout);
        }
        // Support Tablet devices which do not have tab switcher
        else {
            UiObject menu_button = mDevice.findObject(new UiSelector().resourceId(packageID + "menu_button")
                                                              .className("android.widget.ImageButton"));
            menu_button.click();
            newTab = mDevice.findObject(new UiSelector().resourceId(packageID + "menu_item_text")
                                                        .textContains("New tab"));
            newTab.click();
        }
    }

    public void followTextLink(String text) throws Exception {
        UiObject link = mDevice.findObject(new UiSelector().text(text).clickable(true));
        link.waitForExists(uiAutoTimeout);
        link.clickAndWaitForNewWindow();
    }

    @Test
    public void runWorkload() throws Exception {
        // Initial browsing within a single tab
        navigateToPage("https://en.m.wikipedia.org/wiki/Main_Page", true);
        uiDeviceSwipeUp(100);
        sleep(1);
        uiDeviceSwipeUp(100);
        sleep(1);
        uiDeviceSwipeUp(250);
        sleep(1);
        uiDeviceSwipeDown(100);
        navigateToPage("https://en.m.wikipedia.org/wiki/United_States", false);
        uiDeviceSwipeUp(100);
        sleep(1);
        uiDeviceSwipeUp(250);
        sleep(1);
        uiDeviceSwipeDown(100);

        // URL entry and link navigation within a new tab
        newTab();
        navigateToPage("https://en.m.wikipedia.org/wiki/California", true);
        sleep(2);
        followTextLink("United States");
        uiDeviceSwipeDown(50);
        sleep(1);
        uiDeviceSwipeUp(10);
        sleep(3);

        // Pinch to zoom, scroll around
        UiObject webView = mDevice.findObject(new UiSelector().className("android.webkit.WebView"));
        uiObjectVertPinchOut(webView, 100, 50);
        uiDeviceSwipeUp(300);
        sleep(1);
        uiObjectVertPinchIn(webView, 100, 50);
        uiDeviceSwipeUp(100);
        sleep(1);
        uiDeviceSwipeUp(100);
        sleep(3);

        // Go back a page
        pressBack();
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }

    public void runApplicationSetup() throws Exception {
        UiObject sendReportBox;
        UiObject acceptButton, noThanksButton;

        sendReportBox = mDevice.findObject(new UiSelector().resourceId(packageID + "send_report_checkbox")
                                                           .className("android.widget.CheckBox"));
        sendReportBox.click();

        acceptButton = mDevice.findObject(new UiSelector().resourceId(packageID + "terms_accept")
                                                          .className("android.widget.Button"));
        acceptButton.clickAndWaitForNewWindow(uiAutoTimeout);

        noThanksButton = mDevice.findObject(new UiSelector().resourceId(packageID + "negative_button")
                                                            .className("android.widget.Button"));
        noThanksButton.clickAndWaitForNewWindow(uiAutoTimeout);
    }

    public UiObject getLaunchEndObject() {
        UiObject launchEndObject = mDevice.findObject(new UiSelector().className("android.widget.EditText"));
        return launchEndObject;
    }

    public String getLaunchCommand() {
        String launch_command = UiAutoUtils.createLaunchCommand(parameters);
        return launch_command;
    }

    public void setWorkloadParameters(Bundle workload_parameters) {
        parameters = workload_parameters;
        packageID = getPackageID(parameters);
    }
}
