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

import os
import sys
from subprocess import Popen, PIPE

def get_commit():
    p = Popen(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__),
              stdout=PIPE, stderr=PIPE)
    std, _ = p.communicate()
    p.wait()
    if p.returncode:
        return None
    if sys.version_info[0] == 3:
        return std[:8].decode(sys.stdout.encoding, 'replace')
    else:
        return std[:8]
