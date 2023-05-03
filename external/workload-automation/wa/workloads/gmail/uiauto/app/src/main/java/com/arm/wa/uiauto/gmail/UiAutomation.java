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

package com.arm.wa.uiauto.gmail;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.util.Log;

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
    protected String workdir_name;
    protected boolean offlineMode;
    protected String test_image;

    private int networkTimeoutSecs = 30;
    private long networkTimeout =  TimeUnit.SECONDS.toMillis(networkTimeoutSecs);

    @Before
    public void initialize() {
        parameters = getParams();
        packageID = getPackageID(parameters);
        recipient = parameters.getString("recipient");
        workdir_name = parameters.getString("workdir_name");
        offlineMode = parameters.getBoolean("offline_mode");
        test_image = parameters.getString("test_image");
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

            UiObject noEmailAddressMessage = mDevice.findObject(new UiSelector()
                .textContains("Please add at least one email address.")
                .className("android.widget.TextView"));

            if (noEmailAddressMessage.exists()) {
                throw new UiObjectNotFoundException("No email account setup on device. Set up at least one email address");
            }
        }

        // Dismiss fresh new look pop up messages
        UiObject newLookMessageDismissButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "gm_dismiss_button")
                                         .className("android.widget.Button"));
        if(newLookMessageDismissButton.exists()) {
            newLookMessageDismissButton.click();
        }
        //Dismiss secondary message also with same button
        if(newLookMessageDismissButton.exists()) {
            newLookMessageDismissButton.click();
        }

        // Dismiss google meet integration popup
        UiObject googleMeetDismissPopUp =
            mDevice.findObject(new UiSelector().resourceId(packageID + "next_button")
                                         .className("android.widget.Button"));

        if (googleMeetDismissPopUp.exists()) {
            googleMeetDismissPopUp.click();
        }

        // If we're in offline mode we don't need to worry about syncing, so we're done
        if (offlineMode) {
            return;
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

        UiObject conversationView =
            mDevice.findObject(new UiSelector().resourceIdMatches(packageID + "conversation_list.*"));
        if (!conversationView.waitForExists(networkTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"conversationView\".");
        }

        //Get rid of smart compose message on newer versions and return to home screen before ckickNewMail test
        UiObject newMailButton =
            getUiObjectByDescription("Compose");
        newMailButton.click();

        UiObject smartComposeDismissButton = mDevice.findObject(new UiSelector().textContains("Got it")
                                                                                .className("android.widget.Button"));
        if(smartComposeDismissButton.exists()) {
            smartComposeDismissButton.click();
        }

        // Return to conversation/home screen
        mDevice.pressBack();
        if(!conversationView.exists()) {
           mDevice.pressBack();
        }
        if(!conversationView.exists()) {
           mDevice.pressBack();
        }
    }

    public void clickNewMail() throws Exception {
        String testTag = "click_new";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject newMailButton =
            getUiObjectByDescription("Compose");

        logger.start();
        newMailButton.clickAndWaitForNewWindow(uiAutoTimeout);
        logger.stop();
    }

    public void attachImage() throws Exception {
        String testTag = "attach_img";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject attachIcon =
            mDevice.findObject(new UiSelector().resourceId(packageID + "add_attachment"));

        logger.start();

        attachIcon.click();
        UiObject attachFile =
            mDevice.findObject(new UiSelector().textContains("Attach file")
                                               .className("android.widget.TextView"));
        if (!attachFile.exists()){
            attachFile =
                mDevice.findObject(new UiSelector().descriptionContains("Attach file")
                                               .className("android.widget.TextView"));
        }
        attachFile.clickAndWaitForNewWindow(uiAutoTimeout);

        // Show Roots menu
        UiObject rootMenu =
            mDevice.findObject(new UiSelector().descriptionContains("Show root"));
        if (rootMenu.exists()){
            rootMenu.click();
        }

        UiObject imagesEntry =
            mDevice.findObject(new UiSelector().textContains("Images")
                                               .className("android.widget.TextView"));
        if (imagesEntry.waitForExists(uiAutoTimeout)) {
            imagesEntry.click();

            selectGalleryFolder(workdir_name);
            selectGalleryFolder(workdir_name);

            //Switch from grid view to menu view to display filename on larger screens
            UiObject menuListButton = mDevice.findObject(new UiSelector().resourceId("com.android.documentsui:id/menu_list")
                                                                         .className("android.widget.TextView"));
            if (menuListButton.exists()) {
                menuListButton.click();
            }

            UiObject imageButton = mDevice.findObject(new UiSelector().textContains(test_image)
                                                                      .className("android.widget.TextView"));

            imageButton.click();
            imageButton.waitUntilGone(uiAutoTimeout);
        } else { // Use google photos as fallback
            UiObject photos =
                mDevice.findObject(new UiSelector().text("Photos")
                                                   .className("android.widget.TextView"));

            photos.click();

            UiObject working_directory =
                mDevice.findObject(new UiSelector().textContains(workdir_name)
                                                   .className("android.widget.TextView"));

            working_directory.waitForExists (uiAutoTimeout);
            working_directory.click();

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

        UiObject toField = mDevice.findObject(new UiSelector().resourceId(packageID + "to"));
        if (!toField.waitForExists(uiAutoTimeout)) {
            toField = mDevice.findObject(new UiSelector().className("android.widget.EditText"));
        }

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

        UiObject composeField = mDevice.findObject(new UiSelector().textContains("Compose email"));
        if (!composeField.exists()){
            composeField = mDevice.findObject(new UiSelector().descriptionContains("Compose email"));
        }
        if (!composeField.exists()){
            composeField = mDevice.findObject(new UiSelector().resourceId(packageID + "wc_body_layout" ))
                                  .getChild(new UiSelector().className("android.widget.EditText"));
        }

        logger.start();
        composeField.legacySetText("This is a test composition");
        mDevice.pressEnter();
        logger.stop();
    }

    public void clickSendButton() throws Exception {
        String testTag = "click_send";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject sendButton = getUiObjectByDescription("Send");
        logger.start();
        sendButton.clickAndWaitForNewWindow(uiAutoTimeout);
        logger.stop();
        sendButton.waitUntilGone(networkTimeoutSecs);
    }
}
