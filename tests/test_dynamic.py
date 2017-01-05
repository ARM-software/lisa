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


import unittest
import matplotlib
from test_sched import BaseTestSched
from trappy.base import Base
import trappy


class DynamicEvent(Base):

    """Test the ability to register
       specific classes to trappy"""

    unique_word = "dynamic_test_key"
    name = "dynamic_event"


class TestDynamicEvents(BaseTestSched):

    def __init__(self, *args, **kwargs):
        super(TestDynamicEvents, self).__init__(*args, **kwargs)

    def test_dynamic_data_frame(self):
        """
           Test if the dynamic events are populated
           in the data frame
        """
        parse_class = trappy.register_dynamic_ftrace("DynamicEvent", "dynamic_test_key")
        t = trappy.FTrace(name="first")
        self.assertTrue(len(t.dynamic_event.data_frame) == 1)

        trappy.unregister_dynamic_ftrace(parse_class)

    def test_dynamic_class_attr(self):
        """
           Test the attibutes of the dynamically
           generated class
        """
        cls = trappy.register_dynamic_ftrace("DynamicEvent", "dynamic_test_key",
              pivot="test_pivot")
        self.assertEquals(cls.__name__, "DynamicEvent")
        self.assertEquals(cls.name, "dynamic_event")
        self.assertEquals(cls.unique_word, "dynamic_test_key")
        self.assertEquals(cls.pivot, "test_pivot")

        trappy.unregister_dynamic_ftrace(cls)

    def test_dynamic_event_plot(self):
        """Test if plotter can accept a dynamic class
            for a template argument"""

        cls = trappy.register_dynamic_ftrace("DynamicEvent", "dynamic_test_key")
        t = trappy.FTrace(name="first")
        l = trappy.LinePlot(t, cls, column="load")
        l.view(test=True)

        trappy.unregister_dynamic_ftrace(cls)

    def test_dynamic_event_scope(self):
	"""Test the case when an "all" scope class is
	registered. it should appear in both thermal and sched
	ftrace class definitions when scoped ftrace objects are created
	"""
        cls = trappy.register_dynamic_ftrace("DynamicEvent", "dynamic_test_key")
        t1 = trappy.FTrace(name="first")
	self.assertTrue(t1.class_definitions.has_key(cls.name))

        trappy.unregister_dynamic_ftrace(cls)

    def test_register_ftrace_parser(self):
        trappy.register_ftrace_parser(DynamicEvent)
        t = trappy.FTrace(name="first")
        self.assertTrue(len(t.dynamic_event.data_frame) == 1)

        trappy.unregister_ftrace_parser(DynamicEvent)

    def test_no_none_pivot(self):
        """register_dynamic_ftrace() with default value for pivot doesn't create a class with a pivot=None"""
        cls = trappy.register_dynamic_ftrace("MyEvent", "my_dyn_test_key")
        self.assertFalse(hasattr(cls, "pivot"))

        trappy.unregister_dynamic_ftrace(cls)

    def test_unregister_dynamic_ftrace(self):
        """Test that dynamic events can be unregistered"""
        dyn_event = trappy.register_dynamic_ftrace("DynamicEvent",
                                                   "dynamic_test_key")
        trace = trappy.FTrace(name="first")
        self.assertTrue(len(trace.dynamic_event.data_frame) == 1)

        trappy.unregister_dynamic_ftrace(dyn_event)
        trace = trappy.FTrace(name="first")

        self.assertFalse(hasattr(trace, "dynamic_event"))

        dyn_event = trappy.register_dynamic_ftrace("DynamicEvent",
                                                   "dynamic_test_key",
                                                   scope="sched")
        trace = trappy.FTrace(name="first")
        self.assertTrue(len(trace.dynamic_event.data_frame) == 1)

        trappy.unregister_dynamic_ftrace(dyn_event)
        trace = trappy.FTrace(name="first")

        self.assertFalse(hasattr(trace, "dynamic_event"))

    def test_unregister_ftrace_parser(self):
        """unregister_ftrace_parser() works"""
        trappy.register_ftrace_parser(DynamicEvent)
        trappy.unregister_ftrace_parser(DynamicEvent)
        trace = trappy.FTrace()
        self.assertFalse(hasattr(trace, "dynamic_event"))
