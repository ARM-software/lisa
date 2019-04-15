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

#    Copyright 2017 Google, ARM Limited
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

import re
from devlib.module import Module
from devlib.exception import TargetStableError
from devlib.utils.misc import memoized

class GpufreqModule(Module):

    name = 'gpufreq'
    path = ''

    def __init__(self, target):
        super(GpufreqModule, self).__init__(target)
        frequencies_str = self.target.read_value("/sys/kernel/gpu/gpu_freq_table")
        self.frequencies = list(map(int, frequencies_str.split(" ")))
        self.frequencies.sort()
        self.governors = self.target.read_value("/sys/kernel/gpu/gpu_available_governor").split(" ")

    @staticmethod
    def probe(target):
        # kgsl/Adreno
        probe_path = '/sys/kernel/gpu/'
        if target.file_exists(probe_path):
            model = target.read_value(probe_path + "gpu_model")
            if re.search('adreno', model, re.IGNORECASE):
                return True
        return False

    def set_governor(self, governor):
        if governor not in self.governors:
            raise TargetStableError('Governor {} not supported for gpu'.format(governor))
        self.target.write_value("/sys/kernel/gpu/gpu_governor", governor)

    def get_frequencies(self):
        """
        Returns the list of frequencies that the GPU can have
        """
        return self.frequencies

    def get_current_frequency(self):
        """
        Returns the current frequency currently set for the GPU.

        Warning, this method does not check if the gpu is online or not. It will
        try to read the current frequency and the following exception will be
        raised ::

        :raises: TargetStableError if for some reason the frequency could not be read.

        """
        return int(self.target.read_value("/sys/kernel/gpu/gpu_clock"))

    @memoized
    def get_model_name(self):
        """
        Returns the model name reported by the GPU.
        """
        try:
            return self.target.read_value("/sys/kernel/gpu/gpu_model")
        except:  # pylint: disable=bare-except
            return "unknown"
