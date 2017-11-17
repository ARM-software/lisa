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

package com.arm.wa.uiauto.googleplaybooks;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObject2;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiWatcher;
import android.support.test.uiautomator.By;
import android.util.Log;

import com.arm.wa.uiauto.UxPerfUiAutomation.GestureTestParams;
import com.arm.wa.uiauto.UxPerfUiAutomation.GestureType;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.ActionLogger;
import com.arm.wa.uiauto.UiAutoUtils;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.TimeUnit;

import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_DESC;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_ID;
import static com.arm.wa.uiauto.BaseUiAutomation.FindByCriteria.BY_TEXT;


@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation implements ApplaunchInterface {

    private int viewTimeoutSecs = 10;
    private long viewTimeout =  TimeUnit.SECONDS.toMillis(viewTimeoutSecs);

    protected Bundle parameters;
    protected String packageID;

    protected String searchBookTitle;
    protected String libraryBookTitle;
    protected int chapterPageNumber;
    protected String searchWord;
    protected String noteText;

    @Before
    public void initialize() {
        this.uiAutoTimeout = TimeUnit.SECONDS.toMillis(8);

        parameters = getParams();
        packageID = getPackageID(parameters);

        searchBookTitle = parameters.getString("search_book_title");
        libraryBookTitle = parameters.getString("library_book_title");
        chapterPageNumber = parameters.getInt("chapter_page_number");
        searchWord = parameters.getString("search_word");
        noteText = "This is a test note";
    }

    @Test
    public void setup() throws Exception {
        setScreenOrientation(ScreenOrientation.NATURAL);
        runApplicationSetup();

        searchForBook(searchBookTitle);
        addToLibrary();
        openMyLibrary();

        UiWatcher pageSyncPopUpWatcher = createPopUpWatcher();
        registerWatcher("pageSyncPopUp", pageSyncPopUpWatcher);
        runWatchers();
    }
    @Test
    public void runWorkload() throws Exception {
        openBook(libraryBookTitle);
        selectChapter(chapterPageNumber);
        gesturesTest();
        addNote(noteText);
        removeNote();
        searchForWord(searchWord);
        switchPageStyles();
        aboutBook();
        pressBack();
    }

    @Test
    public void teardown() throws Exception {
        removeWatcher("pageSyncPopUp");
        unsetScreenOrientation();
    }

    // Get application parameters and clear the initial run dialogues of the application launch.
    public void runApplicationSetup() throws Exception {
        String account = parameters.getString("account");
        chooseAccount(account);
        clearFirstRunDialogues();
        dismissSendBooksAsGiftsDialog();
        dismissSync();
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

    // Sets the UiObject that marks the end of the application launch.
    public UiObject getLaunchEndObject() {
        UiObject launchEndObject = mDevice.findObject(new UiSelector()
                                         .className("android.widget.TextView")
                                         .textContains("Library"));
        return launchEndObject;
    }

    // If the device has more than one account setup, a prompt appears
    // In this case, select the first account in the list, unless `account`
    // has been specified as a parameter, otherwise select `account`.
    private void chooseAccount(String account) throws Exception {
        UiObject accountPopup =
            mDevice.findObject(new UiSelector().textContains("Choose an account")
                                               .className("android.widget.TextView"));
        if (accountPopup.exists()) {
            if ("None".equals(account)) {
                // If no account has been specified, pick the first entry in the list
                UiObject list =
                    mDevice.findObject(new UiSelector().className("android.widget.ListView"));
                UiObject first = list.getChild(new UiSelector().index(0));
                if (!first.exists()) {
                    // Some devices are not zero indexed. If 0 doesnt exist, pick 1
                    first = list.getChild(new UiSelector().index(1));
                }
                first.click();
            } else {
                // Account specified, select that
                clickUiObject(BY_TEXT, account, "android.widget.CheckedTextView");
            }
            // Click OK to proceed
            UiObject ok =
                mDevice.findObject(new UiSelector().textContains("OK")
                                                   .className("android.widget.Button")
                                                   .enabled(true));
            ok.clickAndWaitForNewWindow();
        }
    }

    // If there is no sample book in My library we are prompted to choose a
    // book the first time application is run. Try to skip the screen or
    // pick a random sample book.
    private void clearFirstRunDialogues() throws Exception {
        UiObject startButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "start_button"));
        // First try and skip the sample book selection
        if (startButton.exists()) {
            startButton.click();
        }

        UiObject endButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "end_button"));
        // Click next button if it exists
        if (endButton.exists()) {
            endButton.click();

            // Select a random sample book to add to My library
            sleep(1);
            tapDisplayCentre();
            sleep(1);

            // Click done button (uses same resource-id)
            endButton.click();
        }
    }

    private void dismissSendBooksAsGiftsDialog() throws Exception {
        UiObject gotIt =
            mDevice.findObject(new UiSelector().textContains("GOT IT"));
        if (gotIt.exists()) {
            gotIt.click();
        }
    }

    private void dismissSync() throws Exception {
        UiObject keepSyncOff =
            mDevice.findObject(new UiSelector().textContains("Keep sync off")
                                               .className("android.widget.Button"));
        if (keepSyncOff.exists()) {
            keepSyncOff.click();
        }
    }

    // Searches for a "free" or "purchased" book title in Google play
    private void searchForBook(final String bookTitle) throws Exception {
        UiObject search =
            mDevice.findObject(new UiSelector().resourceId(packageID + "menu_search"));
        if (!search.exists()) {
            search =
                mDevice.findObject(new UiSelector().resourceId(packageID + "search_box_idle_text"));
        }
        search.click();

        UiObject searchText =
            mDevice.findObject(new UiSelector().textContains("Search")
                                               .className("android.widget.EditText"));
        searchText.setText(bookTitle);
        pressEnter();

        UiObject resultList =
            mDevice.findObject(new UiSelector().resourceId("com.android.vending:id/search_results_list"));
        if (!resultList.waitForExists(viewTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"search results list view\".");
        }

        // Create a selector so that we can search for siblings of the desired
        // book that contains a "free" or "purchased" book identifier
        UiObject label =
            mDevice.findObject(new UiSelector().fromParent(new UiSelector()
                                               .description(String.format("Book: " + bookTitle))
                                               .className("android.widget.TextView"))
                                               .resourceId("com.android.vending:id/li_label")
                                               .descriptionMatches("^(Purchased|Free)$"));

        final int maxSearchTime = 30;
        int searchTime = maxSearchTime;

        while (!label.exists()) {
            if (searchTime > 0) {
                uiDeviceSwipeDown(100);
                sleep(1);
                searchTime--;
            } else {
                throw new UiObjectNotFoundException(
                        "Exceeded maximum search time (" + maxSearchTime  +
                        " seconds) to find book \"" + bookTitle + "\"");
            }
        }

        // Click on either the first "free" or "purchased" book found that
        // matches the book title
        label.click();
    }

    private void addToLibrary() throws Exception {
        UiObject add =
            mDevice.findObject(new UiSelector().textContains("ADD TO LIBRARY")
                                               .className("android.widget.Button"));
        if (add.exists()) {
            // add to My Library and opens book by default
            add.click();
            clickUiObject(BY_TEXT, "BUY", "android.widget.Button", true);
        } else {
            // opens book
            clickUiObject(BY_TEXT, "READ", "android.widget.Button");
        }

        waitForPage();

        UiObject navigationButton =
            mDevice.findObject(new UiSelector().description("Navigate up"));

        // Return to main app window
        pressBack();

        // On some devices screen ordering is not preserved so check for
        // navigation button to determine current screen
        if (navigationButton.exists()) {
            pressBack();
            pressBack();
        }
    }

    private void openMyLibrary() throws Exception {
        String testTag = "open_library";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        logger.start();
        //clickUiObject(BY_DESC, "Show navigation drawer");
        // To correctly find the UiObject we need to specify the index also here
        UiObject myLibrary =
            mDevice.findObject(new UiSelector().className("android.widget.TextView")
                                               .textMatches(".*[lL]ibrary")
                                               .index(3));
        if (!myLibrary.exists()) {
            myLibrary =
                mDevice.findObject(new UiSelector().resourceId(packageID + "jump_text"));
        }
        if (!myLibrary.exists()) {
            myLibrary =
                mDevice.findObject(new UiSelector().resourceId(packageID + "bottom_my_library"));
        }
		myLibrary.clickAndWaitForNewWindow(uiAutoTimeout);

        // Switch to books tab on newer versions
        UiObject books_tab =
            mDevice.findObject(new UiSelector().className("android.widget.TextView")
                                               .textMatches("BOOKS"));
        if (books_tab.exists()){
            books_tab.click();
        }
		logger.stop();
	}

    private void openBook(final String bookTitle) throws Exception {
        String testTag = "open_book";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        long maxWaitTimeSeconds = 120;
        long maxWaitTime = TimeUnit.SECONDS.toMillis(maxWaitTimeSeconds);

        UiSelector bookSelector =
            new UiSelector().text(bookTitle)
                            .className("android.widget.TextView");
        UiObject book = mDevice.findObject(bookSelector);
        // Check that books are sorted by time added to library. This way we
        // can assume any newly downloaded books will be visible on the first
        // screen.
        mDevice.findObject(By.res(packageID + "menu_sort")).click();
        clickUiObject(BY_TEXT, "Recent", "android.widget.TextView");
        // When the book is first added to library it may not appear in
        // cardsGrid until it has been fully downloaded. Wait for fully
        // downloaded books
        UiObject downloadComplete =
            mDevice.findObject(new UiSelector().fromParent(bookSelector)
                                               .description("100% downloaded"));
        if (!downloadComplete.waitForExists(maxWaitTime)) {
                throw new UiObjectNotFoundException(
                        "Exceeded maximum wait time (" + maxWaitTimeSeconds  +
                        " seconds) to download book \"" + bookTitle + "\"");
        }

        logger.start();
        book.click();
        waitForPage();
        logger.stop();
    }

    // Creates a watcher for when a pop up warning appears when pages are out
    // of sync across multiple devices.
    private UiWatcher createPopUpWatcher() throws Exception {
        UiWatcher pageSyncPopUpWatcher = new UiWatcher() {
            @Override
            public boolean checkForCondition() {
                UiObject popUpDialogue =
                    mDevice.findObject(new UiSelector().textStartsWith("You're on page")
                                                       .resourceId("android:id/message"));
                // Don't sync and stay on the current page
                if (popUpDialogue.exists()) {
                    try {
                        UiObject stayOnPage =
                            mDevice.findObject(new UiSelector().text("Yes")
                                                               .className("android.widget.Button"));
                        stayOnPage.click();
                    } catch (UiObjectNotFoundException e) {
                        e.printStackTrace();
                    }
                    return popUpDialogue.waitUntilGone(viewTimeout);
                }
                return false;
            }
        };
        return pageSyncPopUpWatcher;
    }

    private void selectChapter(final int chapterPageNumber) throws Exception {
        getDropdownMenu();

        UiObject contents = getUiObjectByResourceId(packageID + "menu_reader_toc");
        contents.clickAndWaitForNewWindow(uiAutoTimeout);
        UiObject toChapterView = getUiObjectByResourceId(packageID + "toc_list_view",
                                                         "android.widget.ExpandableListView");
        // Navigate to top of chapter view
        searchPage(toChapterView, 1, Direction.UP, 10);
        // Search for chapter page number
        UiObject page = searchPage(toChapterView, chapterPageNumber, Direction.DOWN, 10);
        // Go to the page
        page.clickAndWaitForNewWindow(viewTimeout);

        waitForPage();
    }

    private void gesturesTest() throws Exception {
        String testTag = "gesture";

        // Perform a range of swipe tests while browsing home photoplaybooks gallery
        LinkedHashMap<String, GestureTestParams> testParams = new LinkedHashMap<String, GestureTestParams>();
        testParams.put("swipe_left", new GestureTestParams(GestureType.UIDEVICE_SWIPE, Direction.LEFT, 20));
        testParams.put("swipe_right", new GestureTestParams(GestureType.UIDEVICE_SWIPE, Direction.RIGHT, 20));
        testParams.put("pinch_out", new GestureTestParams(GestureType.PINCH, PinchType.OUT, 100, 50));
        testParams.put("pinch_in", new GestureTestParams(GestureType.PINCH, PinchType.IN, 100, 50));

        Iterator<Entry<String, GestureTestParams>> it = testParams.entrySet().iterator();

        while (it.hasNext()) {
            Map.Entry<String, GestureTestParams> pair = it.next();
            GestureType type = pair.getValue().gestureType;
            Direction dir = pair.getValue().gestureDirection;
            PinchType pinch = pair.getValue().pinchType;
            int steps = pair.getValue().steps;
            int percent = pair.getValue().percent;

            String runName = String.format(testTag + "_" + pair.getKey());
            ActionLogger logger = new ActionLogger(runName, parameters);

            UiObject pageView = waitForPage();

            logger.start();

            switch (type) {
                case UIDEVICE_SWIPE:
                    uiDeviceSwipe(dir, steps);
                    break;
                case PINCH:
                    uiObjectVertPinch(pageView, pinch, steps, percent);
                    break;
                default:
                    break;
            }

            logger.stop();
        }

        waitForPage();
    }

    private void addNote(final String text) throws Exception {
        String testTag = "note_add";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        hideDropDownMenu();

        UiObject clickable = mDevice.findObject(new UiSelector().longClickable(true));
        if (!clickable.exists()){
            clickable = mDevice.findObject(new UiSelector().resourceIdMatches(".*/main_page"));
        }
        if (!clickable.exists()){
            clickable = mDevice.findObject(new UiSelector().resourceIdMatches(".*/reader"));
        }

        logger.start();

        uiObjectPerformLongClick(clickable, 100);

        UiObject addNoteButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "add_note_button"));
        addNoteButton.click();

        UiObject noteEditText = getUiObjectByResourceId(packageID + "note_edit_text",
                                                        "android.widget.EditText");
        noteEditText.setText(text);

        clickUiObject(BY_ID, packageID + "note_menu_button", "android.widget.ImageButton");
        clickUiObject(BY_TEXT, "Save", "android.widget.TextView");

        logger.stop();

        waitForPage();
    }

    private void removeNote() throws Exception {
        String testTag = "note_remove";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        UiObject clickable = mDevice.findObject(new UiSelector().longClickable(true));
        if (!clickable.exists()){
            clickable = mDevice.findObject(new UiSelector().resourceIdMatches(".*/main_page"));
        }
        if (!clickable.exists()){
            clickable = mDevice.findObject(new UiSelector().resourceIdMatches(".*/reader"));
        }

        logger.start();

        uiObjectPerformLongClick(clickable, 100);

        UiObject removeButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "remove_highlight_button"));
        removeButton.click();

        clickUiObject(BY_TEXT, "Remove", "android.widget.Button");

        logger.stop();

        waitForPage();
    }

    private void searchForWord(final String text) throws Exception {
        String testTag = "search_word";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        // Allow extra time for search queries involing high freqency words
        final long searchTimeout =  TimeUnit.SECONDS.toMillis(20);

        getDropdownMenu();

        UiObject search =
            mDevice.findObject(new UiSelector().resourceId(packageID + "menu_search"));
        search.click();

        UiObject searchText =
            mDevice.findObject(new UiSelector().resourceId(packageID + "search_src_text"));

        logger.start();

        searchText.setText(text);
        pressEnter();

        UiObject resultList =
            mDevice.findObject(new UiSelector().resourceId(packageID + "search_results_list"));
        if (!resultList.waitForExists(searchTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"search results list view\".");
        }

        UiObject searchWeb =
            mDevice.findObject(new UiSelector().textMatches("Search web|SEARCH WEB")
                                         .className("android.widget.TextView"));
        if (!searchWeb.waitForExists(searchTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"Search web view\".");
        }

        logger.stop();

        pressBack();
    }

    private void switchPageStyles() throws Exception {
        String testTag = "style";

        getDropdownMenu();

        clickUiObject(BY_ID, packageID + "menu_reader_settings", "android.widget.TextView");

        // Check for lighting option button on newer versions
        UiObject lightingOptionsButton =
            mDevice.findObject(new UiSelector().resourceId(packageID + "lighting_options_button"));
        if (lightingOptionsButton.exists()) {
            lightingOptionsButton.click();
        }

        String[] styles = {"Night", "Sepia", "Day"};
        for (String style : styles) {
            try {
                ActionLogger logger = new ActionLogger(testTag + "_" + style, parameters);
                UiObject pageStyle =
                    mDevice.findObject(new UiSelector().description(style));

                logger.start();
                pageStyle.clickAndWaitForNewWindow(viewTimeout);
                logger.stop();

            } catch (UiObjectNotFoundException e) {
                // On some devices the lighting options menu disappears
                // between clicks. Searching for the menu again would affect
                // the logger timings so log a message and continue
                Log.e("GooglePlayBooks", "Could not find pageStyle \"" + style + "\"");
            }
        }

        sleep(2);
        tapDisplayCentre(); // exit reader settings dialog
        waitForPage();
    }

    private void aboutBook() throws Exception {
        String testTag = "open_about";
        ActionLogger logger = new ActionLogger(testTag, parameters);

        getDropdownMenu();

        clickUiObject(BY_DESC, "More options", "android.widget.ImageView");

        UiObject bookInfo = getUiObjectByText("About this book", "android.widget.TextView");

        logger.start();

        bookInfo.clickAndWaitForNewWindow(uiAutoTimeout);

        UiObject detailsPanel =
            mDevice.findObject(new UiSelector().resourceId("com.android.vending:id/item_details_panel"));
        waitObject(detailsPanel, viewTimeoutSecs);

        logger.stop();

        pressBack();
    }

    // Helper for waiting on a page between actions
    private UiObject waitForPage() throws Exception {
        UiObject activityReader =
            mDevice.findObject(new UiSelector().resourceId(packageID + "activity_reader")
                                               .childSelector(new UiSelector()
                                               .focusable(true)));
        // On some devices the object in the view hierarchy is found before it
        // becomes visible on the screen. Therefore add pause instead.
        sleep(3);

        if (!activityReader.waitForExists(viewTimeout)) {
            throw new UiObjectNotFoundException("Could not find \"activity reader view\".");
        }

        return activityReader;
    }

    // Helper for accessing the drop down menu
    private void getDropdownMenu() throws Exception {
        UiObject actionBar =
            mDevice.findObject(new UiSelector().resourceId(packageID + "action_bar"));
        if (!actionBar.exists()) {
            tapDisplayCentre();
            sleep(1); // Allow previous views to settle
        }

        UiObject card =
            mDevice.findObject(new UiSelector().resourceId(packageID + "cards")
                                               .className("android.view.ViewGroup"));
        if (card.exists()) {
            // On rare occasions tapping a certain word that appears in the centre
            // of the display will bring up a card to describe the word.
            // (Such as a place will bring a map of its location)
            // In this situation, tap centre to go back, and try again
            // at a different set of coordinates
            int x = (int)(getDisplayCentreWidth() * 0.8);
            int y = (int)(getDisplayCentreHeight() * 0.8);
            while (card.exists()) {
                tapDisplay(x, y);
                sleep(1);
            }

            tapDisplay(x, y);
            sleep(1); // Allow previous views to settle
        }

        if (!actionBar.exists()) {
            throw new UiObjectNotFoundException("Could not find \"action bar\".");
        }
    }

    private void hideDropDownMenu() throws Exception {
        UiObject actionBar =
            mDevice.findObject(new UiSelector().resourceId(packageID + "action_bar"));
        if (actionBar.exists()) {
            tapDisplayCentre();
            sleep(1); // Allow previous views to settle
        }

        if (actionBar.exists()) {
            throw new UiObjectNotFoundException("Could not close \"action bar\".");
        }
    }

    private UiObject searchPage(final UiObject view, final int pagenum, final Direction updown,
                                final int attempts) throws Exception {
        if (attempts <= 0) {
            throw new UiObjectNotFoundException("Could not find \"page number\" after several attempts.");
        }

        UiObject page =
            mDevice.findObject(new UiSelector().description(String.format("page " + Integer.toString(pagenum)))
                                         .className("android.widget.TextView"));
        if (!page.exists()) {
            // Scroll up by swiping down
            if (updown == Direction.UP) {
                view.swipeDown(200);
            // Default case is to scroll down (swipe up)
            } else {
                view.swipeUp(200);
            }
            page = searchPage(view, pagenum, updown, attempts - 1);
        }
        return page;
    }
}
