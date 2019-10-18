/*    Copyright 2013-2018 ARM Limited
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

package com.arm.wa.uiauto;

import android.os.Bundle;
import android.os.SystemClock;
import android.app.Instrumentation;
import android.content.Context;
import android.graphics.Point;
import android.graphics.Rect;

import android.support.test.InstrumentationRegistry;
import android.support.test.uiautomator.UiDevice;
import android.support.test.uiautomator.UiObject;
import android.support.test.uiautomator.UiObjectNotFoundException;
import android.support.test.uiautomator.UiSelector;
import android.support.test.uiautomator.UiWatcher;
import android.support.test.uiautomator.UiScrollable;

import org.junit.Before;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static android.support.test.InstrumentationRegistry.getArguments;

public class BaseUiAutomation {

    public enum FindByCriteria { BY_ID, BY_TEXT, BY_DESC };
    public enum Direction { UP, DOWN, LEFT, RIGHT, NULL };
    public enum ScreenOrientation { RIGHT, NATURAL, LEFT, PORTRAIT, LANDSCAPE };
    public enum PinchType { IN, OUT, NULL };

    // Time in milliseconds
    public long uiAutoTimeout = 4 * 1000;

    public static final int CLICK_REPEAT_INTERVAL_MINIMUM = 5;
    public static final int CLICK_REPEAT_INTERVAL_DEFAULT = 50;

    public Instrumentation mInstrumentation;
    public Context mContext;
    public UiDevice mDevice;

    @Before
    public void initialize_instrumentation() {
        mInstrumentation = InstrumentationRegistry.getInstrumentation();
        mDevice = UiDevice.getInstance(mInstrumentation);
        mContext = mInstrumentation.getTargetContext();
    }

    @Test
    public void setup() throws Exception {
    }

    @Test
    public void runWorkload() throws Exception {
    }

    @Test
    public void extractResults() throws Exception {
    }

    @Test
    public void teardown() throws Exception {
    }

    public void sleep(int second) {
        SystemClock.sleep(second * 1000);
    }

    // Generate a package ID
    public String getPackageID(Bundle parameters) {
        String packageName = parameters.getString("package_name");
        return packageName + ":id/";
    }

    public boolean takeScreenshot(String name) {
        Bundle params = getArguments();
        String png_dir = params.getString("workdir");

        try {
            return mDevice.takeScreenshot(new File(png_dir, name + ".png"));
        } catch (NoSuchMethodError e) {
            return true;
        }
    }

    public void waitText(String text) throws UiObjectNotFoundException {
        waitText(text, 600);
    }

    public void waitText(String text, int second) throws UiObjectNotFoundException {
        UiSelector selector = new UiSelector();
        UiObject text_obj = mDevice.findObject(selector.text(text)
                                                       .className("android.widget.TextView"));
        waitObject(text_obj, second);
    }

    public void waitObject(UiObject obj) throws UiObjectNotFoundException {
        waitObject(obj, 600);
    }

    public void waitObject(UiObject obj, int second) throws UiObjectNotFoundException {
        if (!obj.waitForExists(second * 1000)) {
            throw new UiObjectNotFoundException("UiObject is not found: "
                    + obj.getSelector().toString());
        }
    }

    public boolean waitUntilNoObject(UiObject obj, int second) {
        return obj.waitUntilGone(second * 1000);
    }

    public void clearLogcat() throws Exception {
        Runtime.getRuntime().exec("logcat -c");
    }

    public void waitForLogcatText(String searchText, long timeout) throws Exception {
        long startTime = System.currentTimeMillis();
        Process process = Runtime.getRuntime().exec("logcat");
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;

        long currentTime = System.currentTimeMillis();
        boolean found = false;
        while ((currentTime - startTime) < timeout) {
            sleep(2);  // poll every two seconds

            while ((line = reader.readLine()) != null) {
                if (line.contains(searchText)) {
                    found = true;
                    break;
                }
            }

            if (found) {
                break;
            }
            currentTime = System.currentTimeMillis();
        }

        process.destroy();

        if ((currentTime - startTime) >= timeout) {
            throw new TimeoutException("Timed out waiting for Logcat text \"%s\"".format(searchText));
        }
    }

    public void registerWatcher(String name, UiWatcher watcher) {
        mDevice.registerWatcher(name, watcher);
    }

    public void runWatchers() {
        mDevice.runWatchers();
    }

    public void removeWatcher(String name) {
        mDevice.removeWatcher(name);
    }

    public void setScreenOrientation(ScreenOrientation orientation) throws Exception {
        int width = mDevice.getDisplayWidth();
        int height = mDevice.getDisplayHeight();
        switch (orientation) {
            case RIGHT:
                mDevice.setOrientationRight();
                break;
            case NATURAL:
                mDevice.setOrientationNatural();
                break;
            case LEFT:
                mDevice.setOrientationLeft();
                break;
            case LANDSCAPE:
                if (mDevice.isNaturalOrientation()){
                    if (height > width){
                        mDevice.setOrientationRight();
                    }
                }
                else {
                    if (height > width){
                        mDevice.setOrientationNatural();
                    }
                }
                break;
            case PORTRAIT:
                if (mDevice.isNaturalOrientation()){
                    if (height < width){
                        mDevice.setOrientationRight();
                    }
                }
                else {
                    if (height < width){
                        mDevice.setOrientationNatural();
                    }
                }
                break;
            default:
                throw new Exception("No orientation specified");
        }
    }

    public void unsetScreenOrientation() throws Exception {
        mDevice.unfreezeRotation();
    }

    public void uiObjectPerformLongClick(UiObject view, int steps) throws Exception {
        Rect rect = view.getBounds();
        mDevice.swipe(rect.centerX(), rect.centerY(),
                      rect.centerX(), rect.centerY(), steps);
    }

    public int getDisplayHeight() {
        return mDevice.getDisplayHeight();
    }

    public int getDisplayWidth() {
        return mDevice.getDisplayWidth();
    }

    public int getDisplayCentreWidth() {
        return getDisplayWidth() / 2;
    }

    public int getDisplayCentreHeight() {
        return getDisplayHeight() / 2;
    }

    public void tapDisplayCentre() {
        tapDisplay(getDisplayCentreWidth(),  getDisplayCentreHeight());
    }

    public void tapDisplay(int x, int y) {
        mDevice.click(x, y);
    }

    public void pressEnter() {
        mDevice.pressEnter();
    }

    public void pressHome() {
        mDevice.pressHome();
    }

    public void pressBack() {
        mDevice.pressBack();
    }

    public void uiObjectSwipe(UiObject view, Direction direction, int steps) throws Exception {
        switch (direction) {
            case UP:
                view.swipeUp(steps);
                break;
            case DOWN:
                view.swipeDown(steps);
                break;
            case LEFT:
                view.swipeLeft(steps);
                break;
            case RIGHT:
                view.swipeRight(steps);
                break;
            case NULL:
                throw new Exception("No direction specified");
            default:
                break;
        }
    }

    public void uiDeviceSwipeVertical(int startY, int endY, int xCoordinate, int steps) {
        mDevice.swipe(xCoordinate, startY, xCoordinate, endY, steps);
    }

    public void uiDeviceSwipeHorizontal(int startX, int endX, int yCoordinate, int steps) {
        mDevice.swipe(startX, yCoordinate, endX, yCoordinate, steps);
    }

    public void uiObjectVertPinchIn(UiObject view, int steps, int percent) throws Exception {
        final int FINGER_TOUCH_HALF_WIDTH = 20;

        // Make value between 1 and 100
        int nPercent = (percent < 0) ? 1 : (percent > 100) ? 100 : percent;
        float percentage = nPercent / 100f;

        Rect rect = view.getVisibleBounds();

        if (rect.width() <= FINGER_TOUCH_HALF_WIDTH * 2) {
            throw new IllegalStateException("Object width is too small for operation");
        }

        // Start at the top-center and bottom-center of the control
        Point startPoint1 = new Point(rect.centerX(), rect.centerY()
                          + (int) ((rect.height() / 2) * percentage));
        Point startPoint2 = new Point(rect.centerX(), rect.centerY()
                          - (int) ((rect.height() / 2) * percentage));

        // End at the same point at the center of the control
        Point endPoint1 = new Point(rect.centerX(), rect.centerY() + FINGER_TOUCH_HALF_WIDTH);
        Point endPoint2 = new Point(rect.centerX(), rect.centerY() - FINGER_TOUCH_HALF_WIDTH);

        view.performTwoPointerGesture(startPoint1, startPoint2, endPoint1, endPoint2, steps);
    }

    public void uiObjectVertPinchOut(UiObject view, int steps, int percent) throws Exception {
        final int FINGER_TOUCH_HALF_WIDTH = 20;

        // Make value between 1 and 100
        int nPercent = (percent < 0) ? 1 : (percent > 100) ? 100 : percent;
        float percentage = nPercent / 100f;

        Rect rect = view.getVisibleBounds();

        if (rect.width() <= FINGER_TOUCH_HALF_WIDTH * 2) {
            throw new IllegalStateException("Object width is too small for operation");
        }

        // Start from the same point at the center of the control
        Point startPoint1 = new Point(rect.centerX(), rect.centerY() + FINGER_TOUCH_HALF_WIDTH);
        Point startPoint2 = new Point(rect.centerX(), rect.centerY() - FINGER_TOUCH_HALF_WIDTH);

        // End at the top-center and bottom-center of the control
        Point endPoint1 = new Point(rect.centerX(), rect.centerY()
                        + (int) ((rect.height() / 2) * percentage));
        Point endPoint2 = new Point(rect.centerX(), rect.centerY()
                        - (int) ((rect.height() / 2) * percentage));

        view.performTwoPointerGesture(startPoint1, startPoint2, endPoint1, endPoint2, steps);
    }

    public void uiObjectVertPinch(UiObject view, PinchType direction,
                                  int steps, int percent) throws Exception {
        if (direction.equals(PinchType.IN)) {
            uiObjectVertPinchIn(view, steps, percent);
        } else if (direction.equals(PinchType.OUT)) {
            uiObjectVertPinchOut(view, steps, percent);
        }
    }

    public void uiDeviceSwipeUp(int steps) {
        mDevice.swipe(
            getDisplayCentreWidth(),
            (getDisplayCentreHeight() + (getDisplayCentreHeight() / 2)),
            getDisplayCentreWidth(),
            (getDisplayCentreHeight() / 2),
            steps);
    }

    public void uiDeviceSwipeDown(int steps) {
        mDevice.swipe(
            getDisplayCentreWidth(),
            (getDisplayCentreHeight() / 2),
            getDisplayCentreWidth(),
            (getDisplayCentreHeight() + (getDisplayCentreHeight() / 2)),
            steps);
    }

    public void uiDeviceSwipeLeft(int steps) {
        mDevice.swipe(
            (getDisplayCentreWidth() + (getDisplayCentreWidth() / 2)),
            getDisplayCentreHeight(),
            (getDisplayCentreWidth() / 2),
            getDisplayCentreHeight(),
            steps);
    }

    public void uiDeviceSwipeRight(int steps) {
        mDevice.swipe(
            (getDisplayCentreWidth() / 2),
            getDisplayCentreHeight(),
            (getDisplayCentreWidth() + (getDisplayCentreWidth() / 2)),
            getDisplayCentreHeight(),
            steps);
    }

    public void uiDeviceSwipe(Direction direction, int steps) throws Exception {
        switch (direction) {
            case UP:
                uiDeviceSwipeUp(steps);
                break;
            case DOWN:
                uiDeviceSwipeDown(steps);
                break;
            case LEFT:
                uiDeviceSwipeLeft(steps);
                break;
            case RIGHT:
                uiDeviceSwipeRight(steps);
                break;
            case NULL:
                throw new Exception("No direction specified");
            default:
                break;
        }
    }

    public void repeatClickUiObject(UiObject view, int repeatCount, int intervalInMillis) throws Exception {
        int repeatInterval = intervalInMillis > CLICK_REPEAT_INTERVAL_MINIMUM
                             ? intervalInMillis : CLICK_REPEAT_INTERVAL_DEFAULT;
        if (repeatCount < 1 || !view.isClickable()) {
            return;
        }

        for (int i = 0; i < repeatCount; ++i) {
            view.click();
            SystemClock.sleep(repeatInterval); // in order to register as separate click
        }
    }


    public UiObject clickUiObject(FindByCriteria criteria, String matching) throws Exception {
        return clickUiObject(criteria, matching, null, false);
    }

    public UiObject clickUiObject(FindByCriteria criteria, String matching, boolean wait) throws Exception {
        return clickUiObject(criteria, matching, null, wait);
    }

    public UiObject clickUiObject(FindByCriteria criteria, String matching, String clazz) throws Exception {
        return clickUiObject(criteria, matching, clazz, false);
    }

    public UiObject clickUiObject(FindByCriteria criteria, String matching, String clazz, boolean wait) throws Exception {
        UiObject view;

        switch (criteria) {
            case BY_ID:
                view = (clazz == null)
                     ? getUiObjectByResourceId(matching) : getUiObjectByResourceId(matching, clazz);
                break;
            case BY_DESC:
                view = (clazz == null)
                     ? getUiObjectByDescription(matching) : getUiObjectByDescription(matching, clazz);
                break;
            case BY_TEXT:
            default:
                view = (clazz == null)
                     ? getUiObjectByText(matching) : getUiObjectByText(matching, clazz);
                break;
        }

        if (wait) {
            view.clickAndWaitForNewWindow();
        } else {
            view.click();
        }
        return view;
    }

    public UiObject getUiObjectByResourceId(String resourceId, String className) throws Exception {
        return getUiObjectByResourceId(resourceId, className, uiAutoTimeout);
    }

    public UiObject getUiObjectByResourceId(String resourceId, String className, long timeout) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().resourceId(resourceId)
                .className(className));
        if (!object.waitForExists(timeout)) {
            throw new UiObjectNotFoundException(String.format("Could not find \"%s\" \"%s\"",
                    resourceId, className));
        }
        return object;
    }

    public UiObject getUiObjectByResourceId(String id) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().resourceId(id));

        if (!object.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find view with resource ID: " + id);
        }
        return object;
    }

    public UiObject getUiObjectByDescription(String description, String className) throws Exception {
        return getUiObjectByDescription(description, className, uiAutoTimeout);
    }

    public UiObject getUiObjectByDescription(String description, String className, long timeout) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().descriptionContains(description)
                                                             .className(className));
        if (!object.waitForExists(timeout)) {
            throw new UiObjectNotFoundException(String.format("Could not find \"%s\" \"%s\"",
                    description, className));
        }
        return object;
    }

    public UiObject getUiObjectByDescription(String desc) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().descriptionContains(desc));

        if (!object.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find view with description: " + desc);
        }
        return object;
    }

    public UiObject getUiObjectByText(String text, String className) throws Exception {
        return getUiObjectByText(text, className, uiAutoTimeout);
    }

    public UiObject getUiObjectByText(String text, String className, long timeout) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().textContains(text)
                                                             .className(className));
        if (!object.waitForExists(timeout)) {
            throw new UiObjectNotFoundException(String.format("Could not find \"%s\" \"%s\"",
                                                              text, className));
        }
        return object;
    }

    public UiObject getUiObjectByText(String text) throws Exception {
        UiObject object = mDevice.findObject(new UiSelector().textContains(text));

        if (!object.waitForExists(uiAutoTimeout)) {
            throw new UiObjectNotFoundException("Could not find view with text: " + text);
        }
        return object;
    }

    // Helper to select a folder in the gallery
    public void selectGalleryFolder(String directory) throws Exception {
        UiObject workdir =
                mDevice.findObject(new UiSelector().text(directory)
                                                   .className("android.widget.TextView"));
        UiScrollable scrollView =
                new UiScrollable(new UiSelector().scrollable(true));

        // If the folder is not present wait for a short time for
        // the media server to refresh its index.
        boolean discovered = workdir.waitForExists(TimeUnit.SECONDS.toMillis(10));
        if (!discovered && scrollView.exists()) {
            // First check if the directory is visible on the first
            // screen and if not scroll to the bottom of the screen to look for it.
            discovered = scrollView.scrollIntoView(workdir);

            // If still not discovered scroll back to the top of the screen and
            // wait for a longer amount of time for the media server to refresh
            // its index.
            if (!discovered) {
                // scrollView.scrollToBeggining() doesn't work for this
                // particular scrollable view so use device method instead
                for (int i = 0; i < 10; i++) {
                    uiDeviceSwipeUp(20);
                }
                discovered = workdir.waitForExists(TimeUnit.SECONDS.toMillis(60));

                // Scroll to the bottom of the screen one last time
                if (!discovered) {
                    discovered = scrollView.scrollIntoView(workdir);
                }
            }
        }

        if (discovered) {
            workdir.clickAndWaitForNewWindow();
        } else {
            throw new UiObjectNotFoundException("Could not find folder : " + directory);
        }
    }

    // If an an app is not designed for running on the latest version of android
    // (currently Q) dissmiss the warning popup if present.
    public void dismissAndroidVersionPopup() throws Exception {
        UiObject warningText =
            mDevice.findObject(new UiSelector().textContains(
                "This app was built for an older version of Android"));
        UiObject acceptButton =
            mDevice.findObject(new UiSelector().resourceId("android:id/button1")
                                               .className("android.widget.Button"));
        if (warningText.exists() && acceptButton.exists()) {
            acceptButton.click();
        }
    }


    // Override getParams function to decode a url encoded parameter bundle before
    // passing it to workloads.
    public Bundle getParams() {
        // Get the original parameter bundle
        Bundle parameters = getArguments();

        // Decode each parameter in the bundle, except null values and "class", as this
        // used to control instrumentation and therefore not encoded.
        for (String key : parameters.keySet()) {
            String param = parameters.getString(key);
            if (param != null && !key.equals("class")) {
                param = android.net.Uri.decode(param);
                parameters = decode(parameters, key, param);
            }
        }
        return parameters;
    }

    // Helper function to decode a string and insert it as an appropriate type
    // into a provided bundle with its key.
    // Each bundle parameter will be a urlencoded string with 2 characters prefixed to the value
    // used to store the original type information, e.g. 'fl' -> list of floats.
    private Bundle decode(Bundle parameters, String key, String value) {
        char value_type = value.charAt(0);
        char value_dimension = value.charAt(1);
        String param = value.substring(2);

        if (value_dimension == 's') {
            if (value_type == 's') {
                parameters.putString(key, param);
            } else if (value_type == 'f') {
                parameters.putFloat(key, Float.parseFloat(param));
            } else if (value_type == 'd') {
                parameters.putDouble(key, Double.parseDouble(param));
            } else if (value_type == 'b') {
                parameters.putBoolean(key, Boolean.parseBoolean(param));
            } else if (value_type == 'i') {
                parameters.putInt(key, Integer.parseInt(param));
            } else if (value_type == 'n') {
                parameters.putString(key, "None");
            } else {
                throw new IllegalArgumentException("Error decoding:" + key + value
                                                   + " - unknown format");
            }
        } else if (value_dimension == 'l') {
            return decodeArray(parameters, key, value_type, param);
        } else {
            throw new IllegalArgumentException("Error decoding:" + key + value
                    + " - unknown format");
        }
        return parameters;
    }

    // Helper function to deal with decoding arrays and update the bundle with
    // an appropriate array type. The string "0newelement0" is used to distinguish
    // each element from each other in the array when encoded.
    private Bundle decodeArray(Bundle parameters, String key, char type, String value) {
        String[] string_list = value.split("0newelement0");
        if (type == 's') {
            parameters.putStringArray(key, string_list);
        }
        else if (type == 'i') {
            int[] int_list = new int[string_list.length];
            for (int i = 0; i < string_list.length; i++){
                int_list[i] = Integer.parseInt(string_list[i]);
            }
            parameters.putIntArray(key, int_list);
        } else if (type == 'f') {
            float[] float_list = new float[string_list.length];
            for (int i = 0; i < string_list.length; i++){
                float_list[i] = Float.parseFloat(string_list[i]);
            }
            parameters.putFloatArray(key, float_list);
        } else if (type == 'd') {
            double[] double_list = new double[string_list.length];
            for (int i = 0; i < string_list.length; i++){
                double_list[i] = Double.parseDouble(string_list[i]);
            }
            parameters.putDoubleArray(key, double_list);
        } else if (type == 'b') {
            boolean[] boolean_list = new boolean[string_list.length];
            for (int i = 0; i < string_list.length; i++){
                boolean_list[i] = Boolean.parseBoolean(string_list[i]);
            }
            parameters.putBooleanArray(key, boolean_list);
        } else {
            throw new IllegalArgumentException("Error decoding array: " +
                                               value + " - unknown format");
        }
        return parameters;
    }
}
