#    Copyright 2013-2015 ARM Limited
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


import os

from wlauto.core.resource import Resource


class FileResource(Resource):
    """
    Base class for all resources that are a regular file in the
    file system.

    """

    def delete(self, instance):
        os.remove(instance)


class File(FileResource):

    name = 'file'

    def __init__(self, owner, path, url=None):
        super(File, self).__init__(owner)
        self.path = path
        self.url = url

    def __str__(self):
        return '<{}\'s {} {}>'.format(self.owner, self.name, self.path or self.url)


class ExtensionAsset(File):

    name = 'extension_asset'

    def __init__(self, owner, path):
        super(ExtensionAsset, self).__init__(owner, os.path.join(owner.name, path))


class Executable(FileResource):

    name = 'executable'

    def __init__(self, owner, platform, filename):
        super(Executable, self).__init__(owner)
        self.platform = platform
        self.filename = filename

    def __str__(self):
        return '<{}\'s {} {}>'.format(self.owner, self.platform, self.filename)
