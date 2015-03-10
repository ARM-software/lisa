/*    Copyright 2014-2015 ARM Limited
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


package com.arm.wlauto.uiauto.skypevideo;

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

    public static String TAG = "skypevideo";
    public static String videoCallButtonResourceId = "com.skype.raider:id/chat_menu_item_call_video";
    public static String noContactMessage = "Could not find contact \"%s\" in the contacts list.";

    public void runUiAutomation() throws Exception {
            Bundle parameters = getParams();
            String contactName = parameters.getString("name").replace('_', ' ');
            int duration = Integer.parseInt(parameters.getString("duration"));

            selectContact(contactName);
            initiateCall(duration);
    }

    public void selectContact(String name) throws Exception {
            UiSelector selector = new UiSelector();
            UiObject peopleTab = new UiObject(selector.text("People"));
            peopleTab.click();
            sleep(1);  // tab transition

            // Note: this assumes that the contact is in view and does not attempt to scroll to find it.
            // The expectation is that this automation will be used with a dedicated account that was set 
            // up for the purpose and so would only have the intended target plus one or two other contacts 
            // at most  in the list. If that is not the case, then this needs to be re-written to scroll to 
            // find the contact if necessary.
            UiObject contactCard = new UiObject(selector.text(name));
            if (!contactCard.exists()) {
                    throw new UiObjectNotFoundException(String.format(noContactMessage, name));
            }
            contactCard.clickAndWaitForNewWindow();
    }

    public void initiateCall(int duration) throws Exception {
            UiSelector selector = new UiSelector();
            UiObject videoCallButton = new UiObject(selector.resourceId(videoCallButtonResourceId));
            videoCallButton.click();
            sleep(duration);
    }
}
