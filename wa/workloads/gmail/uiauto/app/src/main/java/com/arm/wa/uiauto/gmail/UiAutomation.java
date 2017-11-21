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

package com.arm.wa.uiauto.gmail;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;

import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ActionLogger;
import com.arm.wa.uiauto.UiAutoUtils;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.concurrent.TimeUnit;

@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation implements ApplaunchInterface {

    protected Bundle parameters;
    protected String packageID;
    protected String recipient;

    private int networkTimeoutSecs = 30;
    private long networkTimeout =  TimeUnit.SECONDS.toMillis(networkTimeoutSecs);

    @Before
    public void initialize() {
        parameters = getParams();
        packageID = getPackageID(parameters);
        recipient = parameters.getString("recipient");
    }

    @Test
    public void setup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        runApplicationSetup();
    }

    @Test
    public void runWorkload() throws Exception {
        clickNewMail();
        attachImage();
        setToField(recipient);
        setSubjectField();
        setComposeField();
        clickSendButton();
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }

    public void runApplicationSetup() throws Exception {
        clearFirstRunDialogues();
    }

    // Sets the UiObject that marks the end of the application launch.
    public UiObject getLaunchEndObject() {
        UiObject launchEndObject =
                        mDevice.findObject(new UiSelector().className("android.widget.ImageButton"));
        return launchEndObject;
    }

    // Returns the launch command for the application.
    public String getLaunchCommand() {
        String launch_command;
        launch_command = UiAutoUtils.createLaunchCommand(parameters);
        return launch_command;
    }

    // Pass the workload parameters, used for applaunch
    public void setWorkloadParameters(Bundle workload_parameters) {
        parameters = workload_parameters;
        packageID = getPackageID(parameters);
    }

    public void clearFirstRunDialogues() throws Exception {
        // The first run dialogues vary on different devices so check if they are there and dismiss
        UiObject gotItBox =
            mDevice.findObject(new UiSelector().resourceId(packageID + "welcome_tour_got_it")
                                         .className("android.widget.TextView"));
        if (gotItBox.exists()) {
            gotItBox.clickAndWaitForNewWindow(uiAutoTimeout);
        }

        UiObject takeMeToBox =
            mDevice.findObject(new UiSelector().textContains("Take me to Gmail")
                                         .className("android.widget.TextView"));
        if (takeMeToBox.exists()) {
            takeMeToBox.clickAndWaitForNewWindow(uiAutoTimeout);
        }

        UiObject syncNowButton =
            mDevice.findObject(new UiSelector().textContains("Sync now")
                                         .className("android.widget.Button"));
        if (syncNowButton.exists()) {
            syncNowButton.clickAndWaitForNewWindow(uiAutoTimeout);
            // On some devices we need to wait for a sync to occur after clearing the data
            // We also need to sleep here since waiting for a new window is not enough
            sleep(10);
        }

        // Wait an obnoxiously long period of time for the sync operation to finish
        // If it still fails, then there is a problem with the app obtaining the data it needs
        // Recommend restarting the phone and/or clearing the app data
        UiObject gettingMessages =
            mDevice.findObject(new UiSelector().textContains("Getting your messages")
                                               .className("android.widget.TextView"));
        UiObject waitingSync =
            mDevice.findObject(new UiSelector().textContains("Waiting for sync")
                                               .className("android.widget.TextView"));
        if (!waitUntilNoObject(gettingMessages, networkTimeoutSecs*4) ||
            !waitUntilNoObject(waitingSync, networkTimeoutSecs*4)) {
            throw new UiObjectNotFoundException("Device cannot sync! Try rebooting or clearing app data");
        }
    }

    public void clickNewMail() throws Exception {
        String testTag = "click_new";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject conversationView =
            mDevice.findObject(new UiSelector().resourceIdMatches(packageID + "conversation_list.*"));
        if (!conversationView.waitForExists(networkTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"conversationView\".");
        }

        UiObject newMailButton =
            getUiObjectByDescription("Compose", "android.widget.ImageButton");
        logger.start();
        newMailButton.clickAndWaitForNewWindow(uiAutoTimeout);
        logger.stop();
    }

    public void attachImage() throws Exception {
        String testTag = "attach_img";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject attachIcon =
            getUiObjectByResourceId(packageID + "add_attachment", "android.widget.TextView");

        logger.start();

        attachIcon.click();
        UiObject attachFile =
            getUiObjectByText("Attach file", "android.widget.TextView");
        attachFile.clickAndWaitForNewWindow(uiAutoTimeout);

        // Show Roots menu
        UiObject rootMenu =
            mDevice.findObject(new UiSelector().descriptionContains("Show roots")
                                               .className("android.widget.ImageButton"));
        if (rootMenu.exists()){
            rootMenu.click();
        }
        // Check for Photos
        UiObject photos =
            mDevice.findObject(new UiSelector().text("Photos")
                                               .className("android.widget.TextView"));
        // If Photos does not exist use the images folder
        if (!photos.waitForExists (uiAutoTimeout)) {
            UiObject imagesEntry =
                mDevice.findObject(new UiSelector().textContains("Images")
                                                   .className("android.widget.TextView"));
            if (imagesEntry.waitForExists(uiAutoTimeout)) {
                imagesEntry.click();
            }
            selectGalleryFolder("wa");

            UiObject imageButton =
            mDevice.findObject(new UiSelector().resourceId("com.android.documentsui:id/grid")
                                               .className("android.widget.Gridview")
                                               .childSelector(new UiSelector().index(0)
                                               .className("android.widget.FrameLayout")));
            if (!imageButton.exists()){
                imageButton =
                    mDevice.findObject(new UiSelector().resourceId("com.android.documentsui:id/dir_list")
                                                       .childSelector(new UiSelector().index(0)
                                                       .classNameMatches("android.widget..*Layout")));
                }
            imageButton.click();
            imageButton.waitUntilGone(uiAutoTimeout);
        } else {
            photos.click();
            //Click wa folder image
            UiObject waFolder =
            mDevice.findObject(new UiSelector().textContains("wa")
                                               .className("android.widget.TextView"));
            if (!waFolder.waitForExists (uiAutoTimeout)) {
                UiObject refresh =
                    getUiObjectByResourceId("com.google.android.apps.photos:id/image");
                    refresh.clickAndWaitForNewWindow();
                UiObject back =
                    getUiObjectByResourceId("com.google.android.apps.photos:id/action_mode_close_button");
                    back.clickAndWaitForNewWindow();
            }
            waFolder.waitForExists (uiAutoTimeout);
            waFolder.click();
            //Click test image
            UiObject imageFileButton =
                mDevice.findObject(new UiSelector().descriptionContains("Photo"));
            imageFileButton.click();
            UiObject accept = getUiObjectByText("DONE");
            if (accept.waitForExists (uiAutoTimeout)) {
                accept.click();
            }
        }
        logger.stop();
    }

    public void setToField(String recipient) throws Exception {
        String testTag = "text_to";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject toField = getUiObjectByResourceId(packageID + "to");
        logger.start();
        toField.setText(recipient);
        mDevice.pressEnter();
        logger.stop();
    }

    public void setSubjectField() throws Exception {
        String testTag = "text_subject";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject subjectField = getUiObjectByText("Subject", "android.widget.EditText");
        logger.start();
        // Click on the subject field is required on some platforms to exit the To box cleanly
        subjectField.click();
        subjectField.setText("This is a test message");
        mDevice.pressEnter();
        logger.stop();
    }

    public void setComposeField() throws Exception {
        String testTag = "text_body";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject composeField = mDevice.findObject(new UiSelector().textContains("Compose email")                                                   );
        if (!composeField.exists()){
            composeField = mDevice.findObject(new UiSelector().descriptionContains("Compose email"));
        }

        logger.start();
        composeField.legacySetText("This is a test composition");
        mDevice.pressEnter();
        logger.stop();
    }

    public void clickSendButton() throws Exception {
        String testTag = "click_send";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject sendButton = getUiObjectByDescription("Send", "android.widget.TextView");
        logger.start();
        sendButton.clickAndWaitForNewWindow(uiAutoTimeout);
        logger.stop();
        sendButton.waitUntilGone(networkTimeoutSecs);
    }
}
