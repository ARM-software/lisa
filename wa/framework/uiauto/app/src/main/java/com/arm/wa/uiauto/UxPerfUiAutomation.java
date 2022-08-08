/*    Copyright 2013-2017 ARM Limited
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

import java.util.logging.Logger;

import com.arm.wa.uiauto.BaseUiAutomation.Direction;
import com.arm.wa.uiauto.BaseUiAutomation.PinchType;


public class UxPerfUiAutomation {

    private Logger logger = Logger.getLogger(UxPerfUiAutomation.class.getName());

    public enum GestureType { UIDEVICE_SWIPE, UIOBJECT_SWIPE, PINCH };

    public static class GestureTestParams {
        public GestureType gestureType;
        public Direction gestureDirection;
        public PinchType pinchType;
        public int percent;
        public int steps;

        public GestureTestParams(GestureType gesture, Direction direction, int steps) {
            this.gestureType = gesture;
            this.gestureDirection = direction;
            this.pinchType = PinchType.NULL;
            this.steps = steps;
            this.percent = 0;
        }

        public GestureTestParams(GestureType gesture, PinchType pinchType, int steps, int percent) {
            this.gestureType = gesture;
            this.gestureDirection = Direction.NULL;
            this.pinchType = pinchType;
            this.steps = steps;
            this.percent = percent;
        }
    }
}
