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

package com.arm.wa.uiauto.applaunch;

import android.os.Bundle;
import android.support.test.runner.AndroidJUnit4;
import android.support.test.uiautomator.UiObject;
import android.util.Log;

import com.arm.wa.uiauto.ApplaunchInterface;
import com.arm.wa.uiauto.BaseUiAutomation;
import com.arm.wa.uiauto.UxPerfUiAutomation;
import com.arm.wa.uiauto.ActionLogger;


import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;

import dalvik.system.DexClassLoader;


@RunWith(AndroidJUnit4.class)
public class UiAutomation extends BaseUiAutomation {
    /**
     * Uiobject that marks the end of launch of an application, which is workload
     * specific and added in the workload Java file by a method called getLaunchEndObject().
     */
    public UiObject launchEndObject;
    /** Timeout to wait for application launch to finish. */
    private Integer launch_timeout = 10;
    public String applaunchType;
    public int applaunchIterations;
    public String activityName;
    public ApplaunchInterface launch_workload;

    protected Bundle parameters;
    protected String packageName;
    protected String packageID;

    @Before
    public void initilize() throws Exception {
        parameters = getParams();
        packageID = getPackageID(parameters);

        // Get workload apk file parameters
        String packageName = parameters.getString("package_name");
        String workload = parameters.getString("workload");
        String workloadAPKPath = parameters.getString("workdir");
        String workloadName = String.format("com.arm.wa.uiauto.%1s.apk", workload);
        String workloadAPKFile = String.format("%1s/%2s", workloadAPKPath, workloadName);

        // Load the apk file
        File apkFile = new File(workloadAPKFile);
        File dexLocation = mContext.getDir("outdex", 0);
        if(!apkFile.exists()) {
            throw new Exception(String.format("APK file not found: %s ", workloadAPKFile));
        }
        DexClassLoader classloader = new DexClassLoader(apkFile.toURI().toURL().toString(),
                                                        dexLocation.getAbsolutePath(),
                                                        null, mContext.getClassLoader());

        Class uiautomation = null;
        Object uiautomation_interface = null;
        String workloadClass = String.format("com.arm.wa.uiauto.%1s.UiAutomation", workload);
        try {
            uiautomation = classloader.loadClass(workloadClass);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        Log.d("Class loaded:", uiautomation.getCanonicalName());
        uiautomation_interface = uiautomation.newInstance();

        // Create an Application Interface object from the workload
        launch_workload = ((ApplaunchInterface)uiautomation_interface);
        launch_workload.initialize_instrumentation();
        launch_workload.setWorkloadParameters(parameters);

        // Get parameters for application launch
        applaunchType = parameters.getString("applaunch_type");
        applaunchIterations = parameters.getInt("applaunch_iterations");
        activityName = parameters.getString("launch_activity");
    }

    /**
     * Setup run for applaunch workload that clears the initial
     * run dialogues on launching an application package.
     */
    @Test
    public void setup() throws Exception {
        mDevice.setOrientationNatural();
        launch_workload.runApplicationSetup();
        unsetScreenOrientation();
        closeApplication();
    }

    @Test
    public void runWorkload() throws Exception {
        launchEndObject = launch_workload.getLaunchEndObject();
        for (int iteration = 0; iteration < applaunchIterations; iteration++) {
            Log.d("Applaunch iteration number: ", String.valueOf(applaunchIterations));
            sleep(20);//sleep for a while before next iteration
            killBackground();
            runApplaunchIteration(iteration);
            closeApplication();
        }
    }

    @Test
    public void teardown() throws Exception {
        mDevice.unfreezeRotation();
    }

    /**
     * This method performs multiple iterations of application launch and
     * records the time taken for each iteration.
     */
    public void runApplaunchIteration(Integer iteration_count) throws Exception{
        String testTag = "applaunch" + iteration_count;
        String launchCommand = launch_workload.getLaunchCommand();
        AppLaunch applaunch = new AppLaunch(testTag, launchCommand);
        applaunch.startLaunch();  // Launch the application and start timer
        applaunch.endLaunch();  // marks the end of launch and stops timer
    }

    /*
     * AppLaunch class implements methods that facilitates launching applications
     * from the uiautomator. It has methods that are used for one complete iteration of application
     * launch instrumentation.
     * ActionLogger class is instantiated within the class for measuring applaunch time.
     * startLaunch(): Marks the beginning of the application launch, starts Timer
     * endLaunch(): Marks the end of application, ends Timer
     * launchMain(): Starts the application launch process and validates the finish of launch.
    */
    private class AppLaunch {

        private String testTag;
        private String launchCommand;
        private ActionLogger logger;
        Process launch_p;

        public AppLaunch(String testTag, String launchCommand) {
            this.testTag = testTag;
            this.launchCommand = launchCommand;
            this.logger = new ActionLogger(testTag, parameters);
        }

        // Beginning of application launch
        public void startLaunch() throws Exception{
            logger.start();
            launchMain();
        }

        // Launches the application.
        public void launchMain() throws Exception{
            launch_p = Runtime.getRuntime().exec(launchCommand);
            launchValidate(launch_p);
        }

        // Called by launchMain() to check if app launch is successful
        public void launchValidate(Process launch_p) throws Exception {
            launch_p.waitFor();
            Integer exit_val = launch_p.exitValue();
            if (exit_val != 0) {
                throw new Exception("Application could not be launched");
            }
        }

        // Marks the end of application launch of the workload.
        public void endLaunch() throws Exception{
            waitObject(launchEndObject, launch_timeout);
            logger.stop();
            launch_p.destroy();
        }
    }

    // Exits the application according to application launch type.
    public void closeApplication() throws Exception{
        if(applaunchType.equals("launch_from_background")) {
            pressHome();
        }
        else if(applaunchType.equals("launch_from_long-idle")) {
            killApplication();
            dropCaches();
        }
    }

    // Kills the application process
    public void killApplication() throws Exception{
        Process kill_p;
        String command = String.format("am force-stop %s", packageName);
        kill_p = Runtime.getRuntime().exec(new String[] { "su", "-c", command});
        kill_p.waitFor();
        kill_p.destroy();
    }

    // Kills the background processes
    public void killBackground() throws Exception{
        Process kill_p;
        kill_p = Runtime.getRuntime().exec("am kill-all");
        kill_p.waitFor();
        kill_p.destroy();
    }

    // Drop the caches
    public void dropCaches() throws Exception{
        Process sync;
        sync = Runtime.getRuntime().exec(new String[] { "su", "-c", "sync"});
        sync.waitFor();
        sync.destroy();

        Process drop_cache;
        String command = "echo 3 > /proc/sys/vm/drop_caches";
        drop_cache = Runtime.getRuntime().exec(new String[] { "su", "-c", command});
        drop_cache.waitFor();
        drop_cache.destroy();
    }
}
