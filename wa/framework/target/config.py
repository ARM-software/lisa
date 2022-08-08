#    Copyright 2018 ARM Limited
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

from copy import copy


class TargetConfig(dict):
    """
    Represents a configuration for a target.

    """
    def __init__(self, config=None):
        dict.__init__(self)
        if isinstance(config, TargetConfig):
            self.__dict__ = copy(config.__dict__)
        elif hasattr(config, 'iteritems'):
            for k, v in config.iteritems:
                self.set(k, v)
        elif config:
            raise ValueError(config)

    def set(self, name, value):
        setattr(self, name, value)
