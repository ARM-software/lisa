#    Copyright 2015 ARM Limited
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

import logging

from devlib.utils.types import caseless_string

class CollectorBase(object):

    def __init__(self, target):
        self.target = target
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_path = None

    def reset(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def set_output(self, output_path):
        self.output_path = output_path

    def get_data(self):
        return CollectorOutput()

    def __enter__(self):
        self.reset()
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

class CollectorOutputEntry(object):

    path_kinds = ['file', 'directory']

    def __init__(self, path, path_kind):
        self.path = path

        path_kind = caseless_string(path_kind)
        if path_kind not in self.path_kinds:
            msg = '{} is not a valid path_kind [{}]'
            raise ValueError(msg.format(path_kind, ' '.join(self.path_kinds)))
        self.path_kind = path_kind

    def __str__(self):
        return self.path

    def __repr__(self):
        return '<{} ({})>'.format(self.path, self.path_kind)

    def __fspath__(self):
        """Allow using with os.path operations"""
        return self.path


class CollectorOutput(list):
    pass
