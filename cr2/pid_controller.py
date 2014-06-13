#!/usr/bin/python
"""Process the output of the power allocator's PID controller in the
current directory's trace.dat"""

from thermal import BaseThermal

class PIDController(BaseThermal):
    """Process the power allocator PID controller data in a ftrace dump"""
    def __init__(self, path=None):
        super(PIDController, self).__init__(
            basepath=path,
            unique_word="thermal_power_allocator_pid",
        )
