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

package com.arm.wa.uiauto;

import android.os.Bundle;
import android.util.Log;
    /**
     * Basic marker API for workloads to generate start and end markers for
     * deliminating and timing actions. Markers are output to logcat with debug
     * priority. Actions represent a series of UI interactions to time.
     *
     * The marker API provides a way for instruments and output processors to hook into
     * per-action timings by parsing logcat logs produced per workload iteration.
     *
     * The marker output consists of a logcat tag 'UX_PERF' and a message. The
     * message consists of a name for the action and a timestamp. The timestamp
     * is separated by a single space from the name of the action.
     *
     * Typical usage:
     *
     * ActionLogger logger = ActionLogger("testTag", parameters);
     * logger.start();
     * // actions to be recorded
     * logger.stop();
     */
    public class ActionLogger {

        private String testTag;
        private boolean enabled;

        public ActionLogger(String testTag, Bundle parameters) {
            this.testTag = testTag;
            this.enabled = parameters.getBoolean("markers_enabled");
        }

        public void start() {
            if (enabled) {
                Log.d("UX_PERF", testTag + " start " + System.nanoTime());
            }
        }

        public void stop() throws Exception {
            if (enabled) {
                Log.d("UX_PERF", testTag + " end " + System.nanoTime());
            }
        }
    }
