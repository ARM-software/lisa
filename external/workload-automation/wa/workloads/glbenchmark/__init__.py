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

# pylint: disable=E1101,E0203
import re
import os

from wa import ApkUiautoWorkload, Parameter, Alias
from wa.framework.exception import ConfigError

# These maps provide use-friendly aliases for the most common options.
USE_CASE_MAP = {
    'egypt': 'GLBenchmark 2.5 Egypt HD',
    'egypt-classic': 'GLBenchmark 2.1 Egypt Classic',
    't-rex': 'GLBenchmark 2.7 T-Rex HD',
}

TYPE_MAP = {
    'onscreen': 'C24Z16 Onscreen Auto',
    'offscreen': 'C24Z16 Offscreen Auto',
}


class Glb(ApkUiautoWorkload):

    name = 'glbenchmark'
    description = """
    Measures the graphics performance of Android devices by testing
    the underlying OpenGL (ES) implementation.

    http://gfxbench.com/about-gfxbench.jsp

    From the website:

        The benchmark includes console-quality high-level 3D animations
        (T-Rex HD and Egypt HD) and low-level graphics measurements.

        With high vertex count and complex effects such as motion blur, parallax
        mapping and particle systems, the engine of GFXBench stresses GPUs in order
        provide users a realistic feedback on their device.

    """
    activity = 'com.glbenchmark.activities.GLBenchmarkDownloaderActivity'
    view = 'com.glbenchmark.glbenchmark27/com.glbenchmark.activities.GLBRender'

    package_names = ['com.glbenchmark.glbenchmark27', 'com.glbenchmark.glbenchmark25']
    supported_versions = ['2.7', '2.5']

    # If usecase is not specified the default usecase is the first supported usecase alias
    # for the specified version.
    supported_usecase_aliases = {
        '2.7': ['t-rex', 'egypt'],
        '2.5': ['egypt-classic', 'egypt'],
    }

    default_iterations = 1
    install_timeout = 500
    run_timeout = 4 * 60

    regex = re.compile(r'GLBenchmark (metric|FPS): (.*)')

    parameters = [
        Parameter('version', allowed_values=supported_versions, override=True,
                  description=('Specifies which version of the benchmark to run (different versions '
                               'support different use cases).')),
        Parameter('use_case', default=None,
                  description="""Specifies which usecase to run, as listed in the benchmark menu; e.g.
                                 ``'GLBenchmark 2.5 Egypt HD'``. For convenience, two aliases are provided
                                 for the most common use cases: ``'egypt'`` and ``'t-rex'``. These could
                                 be use instead of the full use case title. For version ``'2.7'`` it defaults
                                 to ``'t-rex'``, for version ``'2.5'`` it defaults to ``'egypt-classic'``.
                  """),
        Parameter('type', default='onscreen',
                  description="""Specifies which type of the use case to run, as listed in the benchmarks
                                 menu (small text underneath the use case name); e.g. ``'C24Z16 Onscreen Auto'``.
                                 For convenience, two aliases are provided for the most common types:
                                 ``'onscreen'`` and ``'offscreen'``. These may be used instead of full type
                                 names.
                  """),
        Parameter('timeout', kind=int, default=200,
                  description="""Specifies how long, in seconds, UI automation will wait for results screen to
                                 appear before assuming something went wrong.
                  """),
    ]

    aliases = [
        Alias('glbench'),
        Alias('egypt', use_case='egypt'),
        Alias('t-rex', use_case='t-rex'),
        Alias('egypt_onscreen', use_case='egypt', type='onscreen'),
        Alias('t-rex_onscreen', use_case='t-rex', type='onscreen'),
        Alias('egypt_offscreen', use_case='egypt', type='offscreen'),
        Alias('t-rex_offscreen', use_case='t-rex', type='offscreen'),
    ]

    def initialize(self, context):
        super(Glb, self).initialize(context)
        self.gui.uiauto_params['version'] = self.version
        if self.use_case is None:
            self.use_case = self.supported_usecase_aliases[self.version][0]
        if self.use_case.lower() in USE_CASE_MAP:
            if self.use_case not in self.supported_usecase_aliases[self.version]:
                raise ConfigError('usecases {} is not supported in version {}'.format(self.use_case, self.version))
            self.use_case = USE_CASE_MAP[self.use_case.lower()]
        self.gui.uiauto_params['use_case'] = self.use_case.replace(' ', '_')

        if self.type.lower() in TYPE_MAP:
            self.type = TYPE_MAP[self.type.lower()]
        self.gui.uiauto_params['usecase_type'] = self.type.replace(' ', '_')

        self.gui.uiauto_params['timeout'] = self.run_timeout

    def update_output(self, context):
        super(Glb, self).update_output(context)
        match_count = 0
        with open(context.get_artifact_path('logcat')) as fh:
            for line in fh:
                match = self.regex.search(line)
                if match:
                    metric = match.group(1)
                    value, units = match.group(2).split()
                    value = value.replace('*', '')
                    if metric == 'metric':
                        metric = 'Frames'
                        units = 'frames'
                    metric = metric + '_' + str(match_count // 2)
                    context.add_metric(metric, value, units)
                    match_count += 1
