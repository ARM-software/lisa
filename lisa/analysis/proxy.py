# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Helper module for registering Analysis classes methods """

import os
import sys
import logging
import inspect
import itertools

from lisa.analysis.base import AnalysisBase
from lisa.utils import Loggable

class AnalysisProxy(Loggable):
    """
    Define list of supported Analysis Classes.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    def __init__(self, trace):
        self.trace = trace
        # Get the list once when the proxy is built, since we know all classes
        # will have had a chance to get registered at that point
        self._class_map = {
            cls.name: cls
            for cls in AnalysisBase.get_subclasses()
            # Classes without a "name" attribute directly defined in their
            # scope will not get registered. That allows having unnamed
            # intermediate base classes that are not meant to be exposed.
            if 'name' in cls.__dict__
        }

        self._instance_map = {}

    def __dir__(self):
        """Provide better completion support for interactive notebook usage"""
        return itertools.chain(super().__dir__(), self._class_map.keys())

    def __getattr__(self, attr):
        # dunder name lookup would have succeeded by now, like __setstate__
        if attr.startswith('__') and attr.endswith('__'):
            return super().__getattribute__(attr)

        logger = self.get_logger()

        # First, try to get the instance of the Analysis that was built if we
        # used it already on that proxy.
        try:
            return self._instance_map[attr]
        except KeyError:
            # If that is the first use, we get the analysis class and build an
            # instance of it
            try:
                analysis_cls = self._class_map[attr]
            except KeyError:
                # No analysis class matching "attr", so we log the ones that
                # are available and let an AttributeError bubble up
                try:
                    analysis_cls = super().__getattribute__(attr)
                except Exception:
                    logger.debug('{} not found. Registered analysis:'.format(attr))
                    for name, cls in list(self._class_map.items()):
                        src_file = '<unknown source>'
                        try:
                            src_file = inspect.getsourcefile(cls) or src_file
                        except TypeError:
                            pass

                        logger.debug('{name} ({cls}) defined in {src}'.format(
                            name=name,
                            cls=cls,
                            src=src_file
                        ))

                    raise
            else:
                analysis_instance = analysis_cls(self.trace)
                self._instance_map[attr] = analysis_instance
                return analysis_instance

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
