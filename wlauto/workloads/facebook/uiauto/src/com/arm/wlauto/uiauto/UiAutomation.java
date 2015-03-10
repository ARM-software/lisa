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


package com.arm.wlauto.uiauto.facebook;

import android.app.Activity;
import android.os.Bundle;
import com.android.uiautomator.core.UiObject;
import com.android.uiautomator.core.UiObjectNotFoundException;
import com.android.uiautomator.core.UiScrollable;
import com.android.uiautomator.core.UiSelector;
import com.android.uiautomator.testrunner.UiAutomatorTestCase;

import com.arm.wlauto.uiauto.BaseUiAutomation;

public class UiAutomation extends BaseUiAutomation {

    public static String TAG = "facebook";

    /*
     * The 'runUiAutomation' method implements the following activities
     * Login to facebook account.
     * Send a message.
     * Check latest notification.
     * Search particular user account and visit his/her facebook account.
     * Go to find friends.
     * Update the facebook status
     */
    public void runUiAutomation() throws Exception {
        final int timeout = 5;
        UiSelector selector = new UiSelector();

        UiObject logInButton = new UiObject(selector
             .className("android.widget.Button").index(3).text("Log In"));

        UiObject emailField = new UiObject(selector
        .className("android.widget.EditText").index(1));
        emailField.clearTextField();
        emailField.setText("abkksathe@gmail.com");

        UiObject passwordField = new UiObject(selector
             .className("android.widget.EditText").index(2));
        passwordField.clearTextField();
        passwordField.setText("highelymotivated");

        logInButton.clickAndWaitForNewWindow(timeout);

        sleep(timeout);

        //Click on message logo
        UiObject messageLogo = new UiObject(new UiSelector()
             .className("android.widget.RelativeLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(3)
             .childSelector(new UiSelector()
             .className("android.widget.RelativeLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.ImageButton").index(0)))));
        messageLogo.clickAndWaitForNewWindow(timeout);

        //send message
        UiObject clickMessage = new UiObject(new UiSelector()
            .className("android.support.v4.view.ViewPager").index(0)
            .childSelector(new UiSelector()
            .className("android.widget.RelativeLayout").index(1)));
        clickMessage.clickAndWaitForNewWindow(timeout);

        sleep(timeout);

        UiObject sendMessage = new UiObject(new UiSelector()
            .className("android.widget.FrameLayout").index(4)
            .childSelector(new UiSelector()
            .className("android.widget.LinearLayout").index(2))
            .childSelector(new UiSelector()
            .className("android.widget.EditText").index(0)
            .text("Write a message")));
        sendMessage.click();

        sleep(timeout);

        UiObject editMessage = new UiObject(new UiSelector()
            .className("android.widget.EditText").text("Write a message"));

        editMessage.setText("Hi how are you?????");

        UiObject sendButton = new UiObject(new UiSelector()
             .className("android.widget.TextView").text("Send"));
        sendButton.click();

        getUiDevice().pressDPadDown();
        sleep(timeout);
        getUiDevice().pressBack();
        sleep(timeout);
        getUiDevice().pressBack();

        //Check for notifications
        UiObject clickNotificationsLogo = new UiObject(new UiSelector()
             .className("android.widget.RelativeLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(3)
             .childSelector(new UiSelector()
             .className("android.widget.RelativeLayout").index(2)
             .childSelector(new UiSelector()
             .className("android.widget.ImageButton").index(0)))));
        clickNotificationsLogo.clickAndWaitForNewWindow(timeout);

        //Click on latest notification
        UiObject clickNotify = new UiObject(new UiSelector()
             .className("android.support.v4.view.ViewPager").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(1)));
        clickNotify.clickAndWaitForNewWindow(timeout);

        sleep(timeout);
        getUiDevice().pressBack();
        sleep(timeout);
        getUiDevice().pressBack();

        //Search for the facebook account
        UiObject clickBar = new UiObject(new UiSelector()
             .className("android.view.View").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.ImageButton").index(0)
             .description("Main navigation menu")));
        clickBar.clickAndWaitForNewWindow(timeout);

        UiObject clickSearch = new UiObject(new UiSelector()
             .className("android.widget.FrameLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.FrameLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.FrameLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.EditText").index(1)
             .text("Search"))))));
        clickSearch.clickAndWaitForNewWindow(timeout);

        UiObject editSearch = new UiObject(new UiSelector()
             .className("android.widget.EditText").index(0).text("Search"));

        editSearch.clearTextField();
        editSearch.setText("amol kamble");
        sleep(timeout);

        UiObject clickOnSearchResult = new UiObject(new UiSelector()
             .className("android.webkit.WebView").index(0));
        clickOnSearchResult.clickTopLeft();

        sleep(2 * timeout);

        getUiDevice().pressBack();
        sleep(timeout);
        getUiDevice().pressBack();

        clickBar.click();

        sleep(timeout);

        //Click on find friends
        UiObject clickFriends = new UiObject(new UiSelector()
             .className("android.widget.FrameLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.FrameLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.FrameLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.RelativeLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.ListView").index(2)))))));

        UiObject friends = clickFriends.getChild(new UiSelector()
             .className("android.widget.RelativeLayout").index(3));
        friends.click();
        sleep(timeout);
        getUiDevice().pressBack();

        //Update the status
        UiObject updateStatus = new UiObject(new UiSelector()
             .className("android.widget.FrameLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.FrameLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.RelativeLayout").index(1)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(0)))))));

        updateStatus.clickAndWaitForNewWindow(timeout);

        UiObject editUpdateStatus = new UiObject(new UiSelector()
             .className("android.widget.EditText")
             .text("What's on your mind?"));
        editUpdateStatus.clearTextField();
        editUpdateStatus.setText("hellllooooooo its done!!");

        UiObject clickPost = new UiObject(new UiSelector()
             .className("android.widget.RelativeLayout").index(0)
             .childSelector(new UiSelector()
             .className("android.widget.LinearLayout").index(3)));
        clickPost.clickAndWaitForNewWindow(timeout);
        getUiDevice().pressHome();
    }

    //disable update using playstore
    public void disableUpdate() throws UiObjectNotFoundException {

        UiObject accountSelect = new UiObject(new UiSelector()
                 .className("android.widget.Button").text("Accept"));

        if (accountSelect.exists())
             accountSelect.click();

        UiObject moreOptions = new UiObject(new UiSelector()
                 .className("android.widget.ImageButton")
                 .description("More options"));
        moreOptions.click();

        UiObject settings = new UiObject(new UiSelector()
                 .className("android.widget.TextView").text("Settings"));
        settings.clickAndWaitForNewWindow();

        UiObject autoUpdate = new UiObject(new UiSelector()
                 .className("android.widget.TextView")
                 .text("Auto-update apps"));

        autoUpdate.clickAndWaitForNewWindow();

        UiObject clickAutoUpdate = new UiObject(new UiSelector()
                  .className("android.widget.CheckedTextView")
                  .text("Do not auto-update apps"));

        clickAutoUpdate.clickAndWaitForNewWindow();

        getUiDevice().pressBack();
        getUiDevice().pressHome();
    }
}
