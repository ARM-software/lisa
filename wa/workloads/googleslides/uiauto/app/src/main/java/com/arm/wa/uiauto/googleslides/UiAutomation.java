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

package com.arm.wa.uiauto.googleslides;

import android.graphics.Rect;
import android.os.Bundle;
import android.os.SystemClock;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.Configurator;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObject2;
import android.support.test.uiautomator.UiScrollable;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.By;

import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.UxPerfUiAutomation;
import com.arm.wa.uiauto.ActionLogger;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_DESC;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_ID;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_TEXT;


@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {

    protected Bundle parameters;
    protected String packageName;
    protected String packageID;

    protected String newDocumentName;
    protected String pushedDocumentName;
    protected int slideCount;
    protected boolean doTextEntry;
    protected String workingDirectoryName;

    public static final int WAIT_TIMEOUT_1SEC = 1000;
    public static final int SLIDE_WAIT_TIME_MS = 200;
    public static final int DEFAULT_SWIPE_STEPS = 10;

    @Before
    public void initialize() {
        parameters = getParams();
        packageID = getPackageID(parameters);

        newDocumentName = parameters.getString("new_doc_name");
        pushedDocumentName = parameters.getString("test_file");
        slideCount = parameters.getInt("slide_count");
        doTextEntry = parameters.getBoolean("do_text_entry");
        workingDirectoryName = parameters.getString("workdir_name");
    }

    @Test
    public void setup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        changeAckTimeout(100);

        skipWelcomeScreen();
        sleep(1);
        dismissUpdateDialog();
        sleep(1);
        dismissWorkOfflineBanner();
        sleep(1);
        enablePowerpointCompat();
        sleep(1);
    }

    @Test
    public void runWorkload() throws Exception {
        testEditNewSlidesDocument(newDocumentName, workingDirectoryName, doTextEntry);
        openDocument(pushedDocumentName, workingDirectoryName);
        waitForProgress(WAIT_TIMEOUT_1SEC*30);
        testSlideshowFromStorage(slideCount);
    }

    @Test
    public void teardown() throws Exception {
        unsetScreenOrientation();
    }

    public void dismissWorkOfflineBanner() throws Exception {
        UiObject banner =
                mDevice.findObject(new UiSelector().textContains("Work offline"));
        if (banner.waitForExists(WAIT_TIMEOUT_1SEC)) {
            clickUiObject(BY_TEXT, "Got it", "android.widget.Button");
        }
    }

    public void dismissUpdateDialog() throws Exception {
        UiObject update =
                mDevice.findObject(new UiSelector().textContains("App update recommended"));
        if (update.waitForExists(WAIT_TIMEOUT_1SEC)) {
            clickUiObject(BY_TEXT, "Dismiss");
        }
    }

    public void enterTextInSlide(String viewName, String textToEnter) throws Exception {
        UiObject view =
                mDevice.findObject(new UiSelector().descriptionMatches(".*[Cc]anvas.*")
                                        .childSelector(new UiSelector()
                                        .descriptionMatches(viewName)));
        view.click();
        mDevice.pressEnter();
        view.legacySetText(textToEnter);

        tapOpenArea();
        // On some devices, keyboard pops up when entering text, and takes a noticeable
        // amount of time (few milliseconds) to disappear after clicking Done.
        // In these cases, trying to find a view immediately after entering text leads
        // to an exception, so a short wait-time is added for stability.
        SystemClock.sleep(SLIDE_WAIT_TIME_MS);
    }

    public void insertSlide(String slideLayout) throws Exception {
        UiObject add_slide =
                mDevice.findObject(new UiSelector().descriptionContains("Add slide"));

        // If we can't see the add slide button the keyboard might still be visiable.
        if (!add_slide.exists()) {
            mDevice.pressBack();
        }
        add_slide.waitForExists(WAIT_TIMEOUT_1SEC);
        add_slide.click();

        UiObject slide_layout = mDevice.findObject(new UiSelector().textContains(slideLayout));

        if (!slide_layout.exists()){
            tapOpenArea();
            UiObject done_button = mDevice.findObject(new UiSelector().resourceId("android:id/action_mode_close_button"));
            if (done_button.exists()){
                done_button.click();
            }
            add_slide.click();
        }
        slide_layout.click();

    }

    public void insertImage(String workingDirectoryName) throws Exception {
        UiObject insertButton = mDevice.findObject(new UiSelector().descriptionContains("Insert"));
        if (insertButton.exists()) {
            insertButton.click();
        } else {
            clickUiObject(BY_DESC, "More options");
            clickUiObject(BY_TEXT, "Insert");
        }
        clickUiObject(BY_TEXT, "Image", true);
        clickUiObject(BY_TEXT, "From photos");

        UiObject imagesFolder = mDevice.findObject(new UiSelector().className("android.widget.TextView").textContains("Images"));
        UiObject moreOptions = mDevice.findObject(new UiSelector().descriptionMatches("More [Oo]ptions"));
        // On some devices the images tabs is missing so we need select the local storage.
        UiObject localDevice = mDevice.findObject(new UiSelector().textMatches(".*[GM]B free"));
        if (!imagesFolder.waitForExists(WAIT_TIMEOUT_1SEC*10)) {
            showRoots();
        }
        if (imagesFolder.exists()) {
            imagesFolder.click();
        } else if (moreOptions.exists()){
            // The local storage can hidden by default so we need to enable showing it.
            moreOptions.click();
            moreOptions.click();
            UiObject internal_storage = mDevice.findObject(new UiSelector().textContains("Show internal storage"));
            if (internal_storage.exists()){
                internal_storage.click();
            }
            mDevice.pressBack();
            showRoots();
        }
        else if (localDevice.exists()){
            localDevice.click();
        }

        UiObject folderEntry = mDevice.findObject(new UiSelector().textContains(workingDirectoryName));
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true));
        if (!folderEntry.exists() && list.waitForExists(WAIT_TIMEOUT_1SEC)) {
            list.scrollIntoView(folderEntry);
        } else {
            folderEntry.waitForExists(WAIT_TIMEOUT_1SEC*10);
        }
        folderEntry.clickAndWaitForNewWindow();

        UiObject picture = mDevice.findObject(new UiSelector().resourceId("com.android.documentsui:id/details"));
        if (!picture.exists()) {
            UiObject pictureAlternate = mDevice.findObject(new UiSelector().resourceId("com.android.documentsui:id/date").enabled(true));
            pictureAlternate.click();
        } else {
            picture.click();
        }
        UiObject done_button = mDevice.findObject(new UiSelector().resourceId("android:id/action_mode_close_button"));
        if (done_button.exists()){
            done_button.click();
        }
    }

    public void insertShape(String shapeName) throws Exception {
        String testTag = "shape_insert";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject insertButton =
                mDevice.findObject(new UiSelector().descriptionContains("Insert"));
        logger.start();
        if (insertButton.exists()) {
            insertButton.click();
        } else {
            clickUiObject(BY_DESC, "More options");
            clickUiObject(BY_TEXT, "Insert");
        }
        clickUiObject(BY_TEXT, "Shape");
        clickUiObject(BY_DESC, shapeName);
        logger.stop();
    }

    public void modifyShape(String shapeName) throws Exception {
        String testTag = "shape_resize";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject resizeHandle =
                mDevice.findObject(new UiSelector().descriptionMatches(".*Bottom[- ]right resize.*"));
        Rect bounds = resizeHandle.getVisibleBounds();
        int newX = bounds.left - 40;
        int newY = bounds.bottom - 40;
        logger.start();
        resizeHandle.dragTo(newX, newY, 40);
        logger.stop();

        testTag = "shape_drag";
        logger = new ActionLogger(testTag, parameters);

        UiObject shapeSelector =
                mDevice.findObject(new UiSelector().resourceId(packageID + "main_canvas")
                        .childSelector(new UiSelector()
                                .descriptionContains(shapeName)));
        logger.start();
        shapeSelector.dragTo(newX, newY, 40);
        logger.stop();
    }

    public void openDocument(String docName, String workingDirectoryName) throws Exception {
        String testTag = "document_open";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        clickUiObject(BY_DESC, "Open presentation");
        clickUiObject(BY_TEXT, "Device storage", true);

        // Allow access to internal storage
        UiObject optionBtn =
            mDevice.findObject(new UiSelector().descriptionContains("More options"));
        if (optionBtn.waitForExists(WAIT_TIMEOUT_1SEC)) {
            optionBtn.click();
            UiObject showInternalBtn =
                mDevice.findObject(new UiSelector().textContains("Show internal storage"));
            // Show internal storage, otherwise already shown so exit menu.
            if (showInternalBtn.exists()) {
                showInternalBtn.click();
            }
            else {
                mDevice.pressBack();
            }
        }
        UiObject workingDirectory = mDevice.findObject(new UiSelector().text(workingDirectoryName));
        UiObject folderEntry = mDevice.findObject(new UiSelector().textContains(workingDirectoryName));

        showRoots();
        UiObject localDevice = mDevice.findObject(new UiSelector().textMatches(".*[GM]B free"));
        localDevice.click();
        UiScrollable list = new UiScrollable(new UiSelector().scrollable(true));
        if (!folderEntry.exists() && list.waitForExists(WAIT_TIMEOUT_1SEC)) {
            list.scrollIntoView(folderEntry);
        } else {
            folderEntry.waitForExists(WAIT_TIMEOUT_1SEC);
        }
        clickUiObject(BY_TEXT, workingDirectoryName);

        UiScrollable fileList =
                new UiScrollable(new UiSelector().className("android.support.v7.widget.RecyclerView"));
        // Older versions of android seem to use a differnt layout
        if (!fileList.waitForExists(WAIT_TIMEOUT_1SEC)) {
            fileList =
                new UiScrollable(new UiSelector().resourceId("com.android.documentsui:id/list"));
        }
        fileList.scrollIntoView(new UiSelector().textContains(docName));

        logger.start();
        clickUiObject(BY_TEXT, docName);
        UiObject open =
            mDevice.findObject(new UiSelector().text("Open"));
        if (open.exists()) {
            open.click();
        }
        logger.stop();
    }

    public void newDocument() throws Exception {
        String testTag = "document_new";
        ActionLogger logger = new ActionLogger(testTag, parameters);
        logger.start();
        clickUiObject(BY_DESC, "New presentation");
        clickUiObject(BY_TEXT, "New PowerPoint", true);
        logger.stop();
        dismissUpdateDialog();
    }

     public void saveDocument(String docName) throws Exception {
       String testTag = "document_save";
       ActionLogger logger = new ActionLogger(testTag, parameters);

       UiObject saveActionButton =
           mDevice.findObject(new UiSelector().textMatches("[Ss]ave|SAVE|"));
       UiObject unsavedIndicator =
           mDevice.findObject(new UiSelector().textContains("Unsaved changes"));
       logger.start();
       if (saveActionButton.waitForExists(WAIT_TIMEOUT_1SEC)) {
           saveActionButton.click();
       } else if (unsavedIndicator.waitForExists(WAIT_TIMEOUT_1SEC)) {
           unsavedIndicator.click();
       }
       clickUiObject(BY_TEXT, "Device");
       UiObject save = clickUiObject(BY_TEXT, "Save", "android.widget.Button");

       // Save in Downloads if present, otherwise assume a sensible defaul location
       UiObject downloadsDir =
            mDevice.findObject(new UiSelector().textContains("Downloads"));
        if (downloadsDir.waitForExists(WAIT_TIMEOUT_1SEC * 5)) {
            downloadsDir.click();
        }

       if (save.waitForExists(WAIT_TIMEOUT_1SEC)) {
           save.click();
       }
       if (saveActionButton.waitForExists(WAIT_TIMEOUT_1SEC)) {
           saveActionButton.click();
       }
       logger.stop();

       // Overwrite if prompted
       // Should not happen under normal circumstances. But ensures test doesn't stop
       // if a previous iteration failed prematurely and was unable to delete the file.
       // Note that this file isn't removed during workload teardown as deleting it is
       // part of the UiAutomator test case.
       UiObject overwriteView =
           mDevice.findObject(new UiSelector().textContains("already exists"));
       if (overwriteView.waitForExists(WAIT_TIMEOUT_1SEC)) {
           clickUiObject(BY_TEXT, "Overwrite");
       }
   }

    public void deleteDocument(String docName) throws Exception {
        String testTag = "document_delete";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        // Switch to Device file tab if present
        UiObject deviceTab =
            mDevice.findObject(new UiSelector().textContains("Device files"));
        if (deviceTab.waitForExists(WAIT_TIMEOUT_1SEC)){
            deviceTab.click();
        }

        UiObject doc =
            mDevice.findObject(new UiSelector().textContains("WORKLOAD"));
        UiObject moreActions =
            doc.getFromParent(new UiSelector().descriptionContains("More actions"));

        logger.start();
        moreActions.click();

        UiObject deleteButton =
                mDevice.findObject(new UiSelector().textMatches(".*([Dd]elete|[Rr]emove).*"));
        if (deleteButton.waitForExists(WAIT_TIMEOUT_1SEC)) {
            deleteButton.click();
        } else {
            // Delete button not found, try to scroll the view
            UiScrollable scrollable =
                    new UiScrollable(new UiSelector().scrollable(true)
                            .childSelector(new UiSelector()
                                    .textMatches(".*(Add people|Save to Drive).*")));
            if (scrollable.exists()) {
                scrollable.scrollIntoView(deleteButton);
            } else {
                UiObject content =
                    mDevice.findObject(new UiSelector().resourceIdMatches(packageID + "(content|menu_recycler_view)"));
                int attemptsLeft = 10; // try a maximum of 10 swipe attempts
                while (!deleteButton.exists() && attemptsLeft > 0) {
                    content.swipeUp(DEFAULT_SWIPE_STEPS);
                    attemptsLeft--;
                }
            }
            deleteButton.click();
        }
        UiObject delete =
                    mDevice.findObject(new UiSelector().textMatches("DELETE|[Dd]elete"));
        if (delete.exists()){
            delete.click();
        }
        delete = mDevice.findObject(new UiSelector().textMatches("MOVE TO BIN"));
        if (delete.exists()){
            delete.click();
        }
        logger.stop();
    }

    protected void skipWelcomeScreen() throws Exception {
        UiObject skip =
            mDevice.findObject(new UiSelector().textMatches("Skip|SKIP"));
        if (skip.exists()) {
            skip.click();
        }
    }

    protected void enablePowerpointCompat() throws Exception {
        String testTag = "enable_pptmode";
        ActionLogger logger = new ActionLogger(testTag, parameters);
        logger.start();

        // Work around to open navigation drawer via swipe.
        uiDeviceSwipeHorizontal(0, getDisplayCentreWidth(), getDisplayCentreHeight() / 2, 10);

        clickUiObject(BY_TEXT, "Settings");
        clickUiObject(BY_TEXT, "Create PowerPoint");
        mDevice.pressBack();
        logger.stop();
    }

    protected void testEditNewSlidesDocument(String docName, String workingDirectoryName, boolean doTextEntry) throws Exception {
        // Init
        newDocument();
        waitForProgress(WAIT_TIMEOUT_1SEC * 30);

        // Slide 1 - Text
        if (doTextEntry) {
            enterTextInSlide(".*[Tt]itle.*", docName);
            windowApplication();
            // Save
            saveDocument(docName);
            sleep(1);
        }

        // Slide 2 - Image
        insertSlide("Title only");
        insertImage(workingDirectoryName);
        sleep(1);

        // If text wasn't entered in first slide, save prompt will appear here
        if (!doTextEntry) {
            // Save
            saveDocument(docName);
            sleep(1);
        }

        // Slide 3 - Shape
        insertSlide("Title slide");
        String shapeName = "Rounded rectangle";
        insertShape(shapeName);
        modifyShape(shapeName);
        mDevice.pressBack();
        UiObject today =
            mDevice.findObject(new UiSelector().text("Today"));
        if (!today.exists()){
            mDevice.pressBack();
        }
        sleep(1);

        // Tidy up
        dismissWorkOfflineBanner(); // if it appears on the homescreen

        // Note: Currently disabled because it fails on Samsung devices
        deleteDocument(docName);
    }

    protected void testSlideshowFromStorage(int slideCount) throws Exception {
        String testTag = "slideshow";
        // Begin Slide show test

        // Note: Using coordinates slightly offset from the slide edges avoids accidentally
        // selecting any shapes or text boxes inside the slides while swiping, which may
        // cause the view to switch into edit mode and fail the test
        UiObject slideCanvas =
                mDevice.findObject(new UiSelector().resourceId(packageID + "main_canvas"));
        Rect canvasBounds = slideCanvas.getVisibleBounds();
        int leftEdge = canvasBounds.left + 10;
        int rightEdge = canvasBounds.right - 10;
        int topEdge = (canvasBounds.top + canvasBounds.bottom) * 1/3 ;
        int bottomEdge = (canvasBounds.top + canvasBounds.bottom) * 2/3 ;

        int yCoordinate = (canvasBounds.top + canvasBounds.bottom) / 2;
        int xCoordinate = (canvasBounds.left + canvasBounds.right) / 2;
        int slideIndex = 0;

        // scroll forward in edit mode
        ActionLogger logger = new ActionLogger(testTag + "_editforward", parameters);
        logger.start();
        while (slideIndex++ < slideCount) {
            uiDeviceSwipeVertical(topEdge, bottomEdge, xCoordinate, DEFAULT_SWIPE_STEPS);
            uiDeviceSwipeHorizontal(rightEdge, leftEdge, yCoordinate, DEFAULT_SWIPE_STEPS);
            waitForProgress(WAIT_TIMEOUT_1SEC*5);
        }
        logger.stop();
        sleep(1);

        // scroll backward in edit mode
        logger = new ActionLogger(testTag + "_editbackward", parameters);
        logger.start();
        while (slideIndex-- > 0) {
            uiDeviceSwipeVertical(bottomEdge, topEdge, xCoordinate, DEFAULT_SWIPE_STEPS);
            uiDeviceSwipeHorizontal(leftEdge, rightEdge, yCoordinate, DEFAULT_SWIPE_STEPS);
            waitForProgress(WAIT_TIMEOUT_1SEC*5);
        }
        logger.stop();
        sleep(1);

        // run slideshow
        UiObject startBtn =
            mDevice.findObject(new UiSelector().descriptionContains("Start slideshow"));
        if (!startBtn.exists()) {
            tapDisplayCentre();
        }

        logger = new ActionLogger(testTag + "_run", parameters);
        logger.start();
        clickUiObject(BY_DESC, "Start slideshow", true);
        UiObject onDevice =
                mDevice.findObject(new UiSelector().textContains("this device"));
        if (onDevice.waitForExists(WAIT_TIMEOUT_1SEC)) {
            onDevice.clickAndWaitForNewWindow();
            waitForProgress(WAIT_TIMEOUT_1SEC*30);
            UiObject presentation =
                    mDevice.findObject(new UiSelector().descriptionContains("Presentation Viewer"));
            presentation.waitForExists(WAIT_TIMEOUT_1SEC*30);
        }
        logger.stop();
        sleep(1);

        slideIndex = 0;

        // scroll forward in slideshow mode
        logger = new ActionLogger(testTag + "_playforward", parameters);
        logger.start();
        while (slideIndex++ < slideCount) {
            uiDeviceSwipeHorizontal(rightEdge, leftEdge, yCoordinate, DEFAULT_SWIPE_STEPS);
            waitForProgress(WAIT_TIMEOUT_1SEC*5);
        }
        logger.stop();
        sleep(1);

        // scroll backward in slideshow mode
        logger = new ActionLogger(testTag + "_playbackward", parameters);
        logger.start();
        while (slideIndex-- > 0) {
            uiDeviceSwipeHorizontal(leftEdge, rightEdge, yCoordinate, DEFAULT_SWIPE_STEPS);
            waitForProgress(WAIT_TIMEOUT_1SEC*5);
        }
        logger.stop();
        sleep(1);

        mDevice.pressBack();
        mDevice.pressBack();
    }

    protected boolean waitForProgress(int timeout) throws Exception {
        UiObject progress = mDevice.findObject(new UiSelector().className("android.widget.ProgressBar"));
        if (progress.waitForExists(WAIT_TIMEOUT_1SEC)) {
            return progress.waitUntilGone(timeout);
        } else {
            return false;
        }
    }

    private long changeAckTimeout(long newTimeout) {
        Configurator config = Configurator.getInstance();
        long oldTimeout = config.getActionAcknowledgmentTimeout();
        config.setActionAcknowledgmentTimeout(newTimeout);
        return oldTimeout;
    }

    private void tapOpenArea() throws Exception {
        UiObject openArea = getUiObjectByResourceId(packageID + "punch_view_pager");
        Rect bounds = openArea.getVisibleBounds();
        // 10px from top of view, 10px from the right edge
        tapDisplay(bounds.right - 10, bounds.top + 10);
    }

    public void windowApplication() throws Exception {
        UiObject window =
                mDevice.findObject(new UiSelector().resourceId("android:id/restore_window"));
        if (window.waitForExists(WAIT_TIMEOUT_1SEC)){
            window.click();
        }
    }

    private void showRoots() throws Exception {
        UiObject rootMenu =
            mDevice.findObject(new UiSelector().descriptionContains("Show root"));
        rootMenu.click();
    }
}
