#    Copyright 2015-2015 ARM Limited
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

    def __init__(self):
        super(DynamicEvent, self).__init__(
            unique_word=self.unique_word,
        )


class TestDynamicEvents(BaseTestSched):

    def __init__(self, *args, **kwargs):
        super(TestDynamicEvents, self).__init__(*args, **kwargs)

    def test_dynamic_data_frame(self):
        """
           Test if the dynamic events are populated
           in the data frame
        """
        trappy.register_dynamic("DynamicEvent", "dynamic_test_key")
        r = trappy.Run(name="first")
        self.assertTrue(len(r.dynamic_event.data_frame) == 1)

    def test_dynamic_class_attr(self):
        """
           Test the attibutes of the dynamically
           generated class
        """
        cls = trappy.register_dynamic("DynamicEvent", "dynamic_test_key",
              pivot="test_pivot")
        self.assertEquals(cls.__name__, "DynamicEvent")
        self.assertEquals(cls.name, "dynamic_event")
        self.assertEquals(cls.unique_word, "dynamic_test_key")
        self.assertEquals(cls.pivot, "test_pivot")

    def test_dynamic_event_plot(self):
        """Test if plotter can accept a dynamic class
            for a template argument"""

        cls = trappy.register_dynamic("DynamicEvent", "dynamic_test_key")
        r = trappy.Run(name="first")
        l = trappy.LinePlot(r, cls, column="load")
        l.view(test=True)

    def test_dynamic_event_scope(self):
	"""Test the case when an "all" scope class is
	registered. it should appear in both thermal and sched
	run class definitions when scoped run objects are created
	"""
        cls = trappy.register_dynamic("DynamicEvent", "dynamic_test_key")
        r1 = trappy.Run(name="first")
	self.assertTrue(r1.class_definitions.has_key(cls.name))

    def test_register_class(self):
        trappy.register_class(DynamicEvent)
        r = trappy.Run(name="first")
        self.assertTrue(len(r.dynamic_event.data_frame) == 1)
