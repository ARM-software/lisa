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

from wlauto import AndroidUiAutoBenchmark, Parameter, Alias
from wlauto.exceptions import ConfigError
import wlauto.common.android.resources

# These maps provide use-friendly aliases for the most common options.
USE_CASE_MAP = {
    'egypt': 'GLBenchmark 2.5 Egypt HD',
    'egypt-classic': 'GLBenchmark 2.1 Egypt Classic',
    't-rex': 'GLBenchmark 2.7 T-Rex HD',
}

VARIANT_MAP = {
    'onscreen': 'C24Z16 Onscreen Auto',
    'offscreen': 'C24Z16 Offscreen Auto',
}


class Glb(AndroidUiAutoBenchmark):

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

    packages = {
        '2.7.0': 'com.glbenchmark.glbenchmark27',
        '2.5.1': 'com.glbenchmark.glbenchmark25',
    }
    # If usecase is not specified the default usecase is the first supported usecase alias
    # for the specified version.
    supported_usecase_aliases = {
        '2.7.0': ['t-rex', 'egypt'],
        '2.5.1': ['egypt-classic', 'egypt'],
    }

    default_iterations = 1
    install_timeout = 500

    regex = re.compile(r'GLBenchmark (metric|FPS): (.*)')

    parameters = [
        Parameter('version', default='2.7.0', allowed_values=['2.7.0', '2.5.1'],
                  description=('Specifies which version of the benchmark to run (different versions '
                               'support different use cases).')),
        Parameter('use_case', default=None,
                  description="""Specifies which usecase to run, as listed in the benchmark menu; e.g.
                                 ``'GLBenchmark 2.5 Egypt HD'``. For convenience, two aliases are provided
                                 for the most common use cases: ``'egypt'`` and ``'t-rex'``. These could
                                 be use instead of the full use case title. For version ``'2.7.0'`` it defaults
                                 to ``'t-rex'``, for version ``'2.5.1'`` it defaults to ``'egypt-classic'``.
                  """),
        Parameter('variant', default='onscreen',
                  description="""Specifies which variant of the use case to run, as listed in the benchmarks
                                 menu (small text underneath the use case name); e.g. ``'C24Z16 Onscreen Auto'``.
                                 For convenience, two aliases are provided for the most common variants:
                                 ``'onscreen'`` and ``'offscreen'``. These may be used instead of full variant
                                 names.
                  """),
        Parameter('times', kind=int, default=1,
                  description=('Specfies the number of times the benchmark will be run in a "tight '
                               'loop", i.e. without performaing setup/teardown inbetween.')),
        Parameter('timeout', kind=int, default=200,
                  description="""Specifies how long, in seconds, UI automation will wait for results screen to
                                 appear before assuming something went wrong.
                  """),
    ]

    aliases = [
        Alias('glbench'),
        Alias('egypt', use_case='egypt'),
        Alias('t-rex', use_case='t-rex'),
        Alias('egypt_onscreen', use_case='egypt', variant='onscreen'),
        Alias('t-rex_onscreen', use_case='t-rex', variant='onscreen'),
        Alias('egypt_offscreen', use_case='egypt', variant='offscreen'),
        Alias('t-rex_offscreen', use_case='t-rex', variant='offscreen'),
    ]

    def __init__(self, device, **kwargs):
        super(Glb, self).__init__(device, **kwargs)
        self.uiauto_params['version'] = self.version

        if self.use_case is None:
            self.use_case = self.supported_usecase_aliases[self.version][0]
        if self.use_case.lower() in USE_CASE_MAP:
            if self.use_case not in self.supported_usecase_aliases[self.version]:
                raise ConfigError('usecases {} is not supported in version {}'.format(self.use_case, self.version))
            self.use_case = USE_CASE_MAP[self.use_case.lower()]
        self.uiauto_params['use_case'] = self.use_case.replace(' ', '_')

        if self.variant.lower() in VARIANT_MAP:
            self.variant = VARIANT_MAP[self.variant.lower()]
        self.uiauto_params['variant'] = self.variant.replace(' ', '_')

        self.uiauto_params['iterations'] = self.times
        self.run_timeout = 4 * 60 * self.times

        self.uiauto_params['timeout'] = self.timeout
        self.package = self.packages[self.version]

    def init_resources(self, context):
        self.apk_file = context.resolver.get(wlauto.common.android.resources.ApkFile(self), version=self.version)
        self.uiauto_file = context.resolver.get(wlauto.common.android.resources.JarFile(self))
        self.device_uiauto_file = self.device.path.join(self.device.working_directory,
                                                        os.path.basename(self.uiauto_file))
        if not self.uiauto_package:
            self.uiauto_package = os.path.splitext(os.path.basename(self.uiauto_file))[0]

    def update_result(self, context):
        super(Glb, self).update_result(context)
        match_count = 0
        with open(self.logcat_log) as fh:
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
                    context.result.add_metric(metric, value, units)
                    match_count += 1

