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
import logging

from wa.framework.plugin import TargetedPlugin
from wa.framework.resource import JarFile, ReventFile, NO_ONE
from wa.framework.exception import WorkloadError

from devlib.utils.android import ApkInfo


class Workload(TargetedPlugin):
    """
    This is the base class for the workloads executed by the framework.
    Each of the methods throwing NotImplementedError *must* be implemented
    by the derived classes.
    """

    kind = 'workload'

    def init_resources(self, context):
        """
        This method may be used to perform early resource discovery and
        initialization. This is invoked during the initial loading stage and
        before the device is ready, so cannot be used for any device-dependent
        initialization. This method is invoked before the workload instance is
        validated.

        """
        pass

    def initialize(self, context):
        """
        This method should be used to perform once-per-run initialization of a
        workload instance, i.e., unlike ``setup()`` it will not be invoked on
        each iteration.
        """
        pass

    def setup(self, context):
        """
        Perform the setup necessary to run the workload, such as copying the
        necessary files to the device, configuring the environments, etc.

        This is also the place to perform any on-device checks prior to
        attempting to execute the workload.
        """
        pass

    def run(self, context):
        """
        Execute the workload. This is the method that performs the actual
        "work" of the.
        """
        pass

    def extract_results(self, context):
        """
        Extract results on the target
        """
        pass

    def update_output(self, context):
        """
        Update the output within the specified execution context with the
        metrics and artifacts form this workload iteration.

        """
        pass

    def teardown(self, context):
        """ Perform any final clean up for the Workload. """
        pass

    def finalize(self, context):
        pass

    def __str__(self):
        return '<Workload {}>'.format(self.name)


class UiAutomatorGUI(object):

    def __init__(self, target, package='', klass='UiAutomation',
                 method='runUiAutoamtion'):
        self.target = target
        self.uiauto_package = package
        self.uiauto_class = klass
        self.uiauto_method = method
        self.uiauto_file = None
        self.target_uiauto_file = None
        self.command = None
        self.uiauto_params = {}

    def init_resources(self, context):
        self.uiauto_file = context.resolver.get(JarFile(self))
        self.target_uiauto_file = self.target.path.join(self.target.working_directory,
                                                        os.path.basename(self.uiauto_file))
        if not self.uiauto_package:
            self.uiauto_package = os.path.splitext(os.path.basename(self.uiauto_file))[0]

    def validate(self):
        if not self.uiauto_file:
            raise WorkloadError('No UI automation JAR file found for workload {}.'.format(self.name))
        if not self.uiauto_package:
            raise WorkloadError('No UI automation package specified for workload {}.'.format(self.name))

    def setup(self, context):
        method_string = '{}.{}#{}'.format(self.uiauto_package, self.uiauto_class, self.uiauto_method)
        params_dict = self.uiauto_params
        params_dict['workdir'] = self.target.working_directory
        params = ''
        for k, v in self.uiauto_params.iteritems():
            params += ' -e {} {}'.format(k, v)
        self.command = 'uiautomator runtest {}{} -c {}'.format(self.target_uiauto_file, params, method_string)
        self.target.push_file(self.uiauto_file, self.target_uiauto_file)
        self.target.killall('uiautomator')

    def run(self, context):
        result = self.target.execute(self.command, self.run_timeout)
        if 'FAILURE' in result:
            raise WorkloadError(result)
        else:
            self.logger.debug(result)
        time.sleep(DELAY)

    def teardown(self, context):
        self.target.delete_file(self.target_uiauto_file)


class ReventGUI(object):

    def __init__(self, workload, target, setup_timeout=5 * 60, run_timeout=10 * 60):
        self.workload = workload
        self.target = target
        self.setup_timeout = setup_timeout
        self.run_timeout = run_timeout
        self.on_target_revent_binary = self.target.get_workpath('revent')
        self.on_target_setup_revent = self.target.get_workpath('{}.setup.revent'.format(self.target.name))
        self.on_target_run_revent = self.target.get_workpath('{}.run.revent'.format(self.target.name))
        self.logger = logging.getLogger('revent')
        self.revent_setup_file = None
        self.revent_run_file = None

    def init_resources(self, context):
        self.revent_setup_file = context.resolver.get(ReventFile(self.workload, 'setup'))
        self.revent_run_file = context.resolver.get(ReventFile(self.workload, 'run'))

    def setup(self, context):
        self._check_revent_files(context)
        self.target.killall('revent')
        command = '{} replay {}'.format(self.on_target_revent_binary, self.on_target_setup_revent)
        self.target.execute(command, timeout=self.setup_timeout)

    def run(self, context):
        command = '{} replay {}'.format(self.on_target_revent_binary, self.on_target_run_revent)
        self.logger.debug('Replaying {}'.format(os.path.basename(self.on_target_run_revent)))
        self.target.execute(command, timeout=self.run_timeout)
        self.logger.debug('Replay completed.')

    def teardown(self, context):
        self.target.remove(self.on_target_setup_revent)
        self.target.remove(self.on_target_run_revent)

    def _check_revent_files(self, context):
        # check the revent binary
        revent_binary = context.resolver.get(Executable(NO_ONE, self.target.abi, 'revent'))
        if not os.path.isfile(revent_binary):
            message = '{} does not exist. '.format(revent_binary)
            message += 'Please build revent for your system and place it in that location'
            raise WorkloadError(message)
        if not self.revent_setup_file:
            # pylint: disable=too-few-format-args
            message = '{0}.setup.revent file does not exist, Please provide one for your target, {0}'
            raise WorkloadError(message.format(self.target.name))
        if not self.revent_run_file:
            # pylint: disable=too-few-format-args
            message = '{0}.run.revent file does not exist, Please provide one for your target, {0}'
            raise WorkloadError(message.format(self.target.name))

        self.on_target_revent_binary = self.target.install(revent_binary)
        self.target.push(self.revent_run_file, self.on_target_run_revent)
        self.target.push(self.revent_setup_file, self.on_target_setup_revent)


class ApkHander(object):

    def __init__(self, owner, target, view, install_timeout=300, version=None,
                 strict=True, force_install=False, uninstall=False):
        self.logger = logging.getLogger('apk')
        self.owner = owner
        self.target = target
        self.version = version
        self.apk_file = None
        self.apk_info = None
        self.apk_version = None
        self.logcat_log = None

    def init_resources(self, context):
        self.apk_file = context.resolver.get(ApkFile(self.owner),
                                             version=self.version,
                                             strict=strict)
        self.apk_info = ApkInfo(self.apk_file)

    def setup(self, context):
        self.initialize_package(context)
        self.start_activity()
        self.target.execute('am kill-all')  # kill all *background* activities
        self.target.clear_logcat()

    def initialize_package(self, context):
        installed_version = self.target.get_package_version(self.apk_info.package)
        if self.strict:
            self.initialize_with_host_apk(context, installed_version)
        else:
            if not installed_version:
                message = '''{} not found found on the device and check_apk is set to "False"
                             so host version was not checked.'''
                raise WorkloadError(message.format(self.package))
            message = 'Version {} installed on device; skipping host APK check.'
            self.logger.debug(message.format(installed_version))
            self.reset(context)
            self.version = installed_version

    def initialize_with_host_apk(self, context, installed_version):
        if installed_version != self.apk_file.version_name:
            if installed_version:
                message = '{} host version: {}, device version: {}; re-installing...'
                self.logger.debug(message.format(os.path.basename(self.apk_file),
                                                 host_version, installed_version))
            else:
                message = '{} host version: {}, not found on device; installing...'
                self.logger.debug(message.format(os.path.basename(self.apk_file),
                                                 host_version))
            self.force_install = True  # pylint: disable=attribute-defined-outside-init
        else:
            message = '{} version {} found on both device and host.'
            self.logger.debug(message.format(os.path.basename(self.apk_file),
                                             host_version))
        if self.force_install:
            if installed_version:
                self.device.uninstall(self.package)
            self.install_apk(context)
        else:
            self.reset(context)
        self.apk_version = host_version

    def start_activity(self):
        output = self.device.execute('am start -W -n {}/{}'.format(self.package, self.activity))
        if 'Error:' in output:
            self.device.execute('am force-stop {}'.format(self.package))  # this will dismiss any erro dialogs
            raise WorkloadError(output)
        self.logger.debug(output)

    def reset(self, context):  # pylint: disable=W0613
        self.device.execute('am force-stop {}'.format(self.package))
        self.device.execute('pm clear {}'.format(self.package))

    def install_apk(self, context):
        output = self.device.install(self.apk_file, self.install_timeout)
        if 'Failure' in output:
            if 'ALREADY_EXISTS' in output:
                self.logger.warn('Using already installed APK (did not unistall properly?)')
            else:
                raise WorkloadError(output)
        else:
            self.logger.debug(output)

    def update_result(self, context):
        self.logcat_log = os.path.join(context.output_directory, 'logcat.log')
        self.device.dump_logcat(self.logcat_log)
        context.add_iteration_artifact(name='logcat',
                                       path='logcat.log',
                                       kind='log',
                                       description='Logact dump for the run.')

    def teardown(self, context):
        self.device.execute('am force-stop {}'.format(self.package))
        if self.uninstall_apk:
            self.device.uninstall(self.package)
