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

from lisa.analysis.base import AnalysisBase

class AnalysisProxy(object):
    """
    Define list of supported Analysis Classes.

    :param trace: input Trace object
    :type trace: :class:`trace.Trace`
    """

    def __init__(self, trace):
        # Get the list once when the proxy is built, since we know all classes
        # will have had a chance to get registered at that point
        self._class_map = {
            cls.name: cls
            for cls in AnalysisBase.get_subclasses()
        }

        self._instance_map = {}

    def __getattr__(self, attr):
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
                    analysis_cls = super(AnalysisProxy, self).__getattribute__(attr)
                except Exception:
                    logger = logging.getLogger('Analysis {} not found'.format(attr))
                    logger.debug('Registered analysis:')
                    for name, cls in self._class_map.items():
                        try:
                            src_file = inspect.getsourcefile(cls)
                        except TypeError:
                            src_file = '<unknown source>'

                        logger.debug('{name} ({cls}) defined in {src}'.format(
                            name=name,
                            cls=cls,
                            src=src_file
                        ))

                    raise

            analysis_instance = analysis_cls(self)
            self._instance_map[attr] = analysis_instance
            return analysis_instance

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
