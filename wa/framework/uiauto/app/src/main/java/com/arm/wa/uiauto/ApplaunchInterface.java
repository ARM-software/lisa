/*    Copyright 2013-2016 ARM Limited
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
import android.support.test.uiautomator.UiObject;

/**
 * ApplaunchInterface.java
 * Interface used for enabling uxperfapplaunch workload.
 * This interface gets implemented by all workloads that support application launch
 * instrumentation.
 */

public interface ApplaunchInterface {

    /**
     * Sets the launchEndObject of a workload, which is a UiObject that marks
     * the end of the application launch.
     */
    public UiObject getLaunchEndObject();

    /**
     * Runs the Uiautomation methods for clearing the initial run
     * dialogues on the first time installation of an application package.
     */
    public void runApplicationSetup() throws Exception;

    /**
     * Provides the application launch command of the application which is
     * constructed as a string from the workload.
     */
    public String getLaunchCommand();

    /** Passes the workload parameters. */
    public void setWorkloadParameters(Bundle parameters);

    /** Initialize the instrumentation for the workload */
    public void initialize_instrumentation();

}
