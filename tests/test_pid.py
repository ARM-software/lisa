#    Copyright 2015-2017 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import matplotlib

from test_thermal import BaseTestThermal
import trappy

class TestPIDController(BaseTestThermal):
    def test_dataframe(self):
        """Test that PIDController() generates a valid data_frame"""
        pid = trappy.FTrace().pid_controller

        self.assertTrue(len(pid.data_frame) > 0)
        self.assertTrue("err_integral" in pid.data_frame.columns)
        self.assertEquals(pid.data_frame["err"].iloc[0], 3225)

    def test_plot_controller(self):
        """Test PIDController.plot_controller()

        As it happens with all plot functions, just test that it doesn't explode"""
        pid = trappy.FTrace().pid_controller

        pid.plot_controller()
        matplotlib.pyplot.close('all')

        pid.plot_controller(title="Antutu", width=20, height=5)
        matplotlib.pyplot.close('all')

        _, ax = matplotlib.pyplot.subplots()
        pid.plot_controller(ax=ax)
        matplotlib.pyplot.close('all')
