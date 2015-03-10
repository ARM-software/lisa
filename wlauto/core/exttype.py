#    Copyright 2014-2015 ARM Limited
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


# Separate module to avoid circular dependencies
from wlauto.core.bootstrap import settings
from wlauto.core.extension import Extension
from wlauto.utils.misc import load_class


_extension_bases = {ext.name: load_class(ext.cls) for ext in settings.extensions}


def get_extension_type(ext):
    """Given an instance of ``wlauto.core.Extension``, return a string representing
    the type of the extension (e.g. ``'workload'`` for a Workload subclass instance)."""
    if not isinstance(ext, Extension):
        raise ValueError('{} is not an instance of Extension'.format(ext))
    for name, cls in _extension_bases.iteritems():
        if isinstance(ext, cls):
            return name
    raise ValueError('Unknown extension type: {}'.format(ext.__class__.__name__))

