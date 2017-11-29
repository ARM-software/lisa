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
import os
import re
import time

from wa import Parameter
from wa.framework.plugin import TargetedPlugin
from wa.framework.resource import (ApkFile, ReventFile,
                                   File, loose_version_matching)
from wa.framework.exception import WorkloadError
from wa.utils.types import ParameterDict
from wa.utils.revent import ReventRecorder
from wa.utils.exec_control import once_per_instance

from devlib.utils.android import ApkInfo


class Workload(TargetedPlugin):
    """
    This is the base class for the workloads executed by the framework.
    Each of the methods throwing NotImplementedError *must* be implemented
    by the derived classes.
    """

    kind = 'workload'

    parameters = [
        Parameter('cleanup_assets', kind=bool,
                  default=True,
                  description="""
                  If ``True``, if assets are deployed as part of the workload they
                  will be removed again from the device as part of finalize.
                  """)
    ]

    # Set this to True to mark that this workload poses a risk of exposing
    # information to the outside world about the device it runs on. An example of
    # this would be a benchmark application that sends scores and device data to a
    # database owned by the maintainer.
    # The user can then set allow_phone_home=False in their configuration to
    # prevent this workload from being run accidentally.
    phones_home = False

    # Set this to ``True`` to mark the the workload will fail without a network
    # connection, this enables it to fail early with a clear message.
    requires_network = False

    # Set this to specify a custom directory for assets to be pushed to, if unset
    # the working directory will be used.
    asset_directory = None

    # Used to store information about workload assets.
    deployable_assets = []

    def __init__(self, target, **kwargs):
        super(Workload, self).__init__(target, **kwargs)
        self.asset_files = []
        self.deployed_assets = []

    def init_resources(self, context):
        """
        This method may be used to perform early resource discovery and
        initialization. This is invoked during the initial loading stage and
        before the device is ready, so cannot be used for any device-dependent
        initialization. This method is invoked before the workload instance is
        validated.

        """
        for asset in self.deployable_assets:
            self.asset_files.append(context.resolver.get(File(self, asset)))

    @once_per_instance
    def initialize(self, context):
        """
        This method should be used to perform once-per-run initialization of a
        workload instance, i.e., unlike ``setup()`` it will not be invoked on
        each iteration.
        """
        if self.asset_files:
            self.deploy_assets(context)

    def setup(self, context):
        """
        Perform the setup necessary to run the workload, such as copying the
        necessary files to the device, configuring the environments, etc.

        This is also the place to perform any on-device checks prior to
        attempting to execute the workload.
        """
        # pylint: disable=unused-argument
        if self.requires_network and not self.target.is_network_connected():
            raise WorkloadError(
                'Workload "{}" requires internet. Target does not appear '
                'to be connected to the internet.'.format(self.name))


    def run(self, context):
        """
        Execute the workload. This is the method that performs the actual
        "work" of the workload.
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

    @once_per_instance
    def finalize(self, context):
        if self.cleanup_assets:
            self.remove_assets(context)

    def deploy_assets(self, context):
        """ Deploy assets if available to the target """
        # pylint: disable=unused-argument
        if not self.asset_directory:
            self.asset_directory = self.target.working_directory
        else:
            self.target.execute('mkdir -p {}'.format(self.asset_directory))

        for asset in self.asset_files:
            self.target.push(asset, self.asset_directory)
            self.deployed_assets.append(self.target.path.join(self.asset_directory,
                                                              os.path.basename(asset)))

    def remove_assets(self, context):
        """ Cleanup assets deployed to the target """
        # pylint: disable=unused-argument
        for asset in self.deployed_assets:
            self.target.remove(asset)

    def __str__(self):
        return '<Workload {}>'.format(self.name)


class ApkWorkload(Workload):

    # May be optionally overwritten by subclasses
    # Times are in seconds
    loading_time = 10
    package_names = []
    view = None

    parameters = [
        Parameter('package_name', kind=str,
                  description="""
                  The package name that can be used to specify
                  the workload apk to use.
                  """),
        Parameter('install_timeout', kind=int,
                  constraint=lambda x: x > 0,
                  default=300,
                  description="""
                  Timeout for the installation of the apk.
                  """),
        Parameter('version', kind=str,
                  default=None,
                  description="""
                  The version of the package to be used.
                  """),
        Parameter('variant', kind=str,
                  default=None,
                  description="""
                  The variant of the package to be used.
                  """),
        Parameter('strict', kind=bool,
                  default=False,
                  description="""
                  Whether to throw an error if the specified package cannot be found
                  on host.
                  """),
        Parameter('force_install', kind=bool,
                  default=False,
                  description="""
                  Always re-install the APK, even if matching version is found already installed
                  on the device.
                  """),
        Parameter('uninstall', kind=bool,
                  default=False,
                  description="""
                  If ``True``, will uninstall workload\'s APK as part of teardown.'
                  """),
        Parameter('exact_abi', kind=bool,
                  default=False,
                  description="""
                  If ``True``, workload will check that the APK matches the target
                  device ABI, otherwise any suitable APK found will be used.
                  """)
    ]

    @property
    def package(self):
        return self.apk.package

    def __init__(self, target, **kwargs):
        super(ApkWorkload, self).__init__(target, **kwargs)
        self.apk = PackageHandler(self,
                                  package_name=self.package_name,
                                  variant=self.variant,
                                  strict=self.strict,
                                  version=self.version,
                                  force_install=self.force_install,
                                  install_timeout=self.install_timeout,
                                  uninstall=self.uninstall,
                                  exact_abi=self.exact_abi)

    @once_per_instance
    def initialize(self, context):
        super(ApkWorkload, self).initialize(context)
        self.apk.initialize(context)
        if self.view is None:
            self.view = 'SurfaceView - {}/{}'.format(self.apk.package,
                                                     self.apk.activity)

    def setup(self, context):
        super(ApkWorkload, self).setup(context)
        self.apk.setup(context)
        time.sleep(self.loading_time)

    def teardown(self, context):
        super(ApkWorkload, self).teardown(context)
        self.apk.teardown()


    def deploy_assets(self, context):
        super(ApkWorkload, self).deploy_assets(context)
        self.target.refresh_files(self.deployed_assets)


class ApkUIWorkload(ApkWorkload):

    def __init__(self, target, **kwargs):
        super(ApkUIWorkload, self).__init__(target, **kwargs)
        self.gui = None

    def init_resources(self, context):
        super(ApkUIWorkload, self).init_resources(context)
        self.gui.init_resources(context.resolver)

    @once_per_instance
    def initialize(self, context):
        super(ApkUIWorkload, self).initialize(context)

    def setup(self, context):
        super(ApkUIWorkload, self).setup(context)
        self.gui.deploy()
        self.gui.setup()

    def run(self, context):
        super(ApkUIWorkload, self).run(context)
        self.gui.run()

    def extract_results(self, context):
        super(ApkUIWorkload, self).extract_results(context)
        self.gui.extract_results()

    def teardown(self, context):
        self.gui.teardown()
        super(ApkUIWorkload, self).teardown(context)

    @once_per_instance
    def finalize(self, context):
        super(ApkUIWorkload, self).finalize(context)
        self.gui.remove()


class ApkUiautoWorkload(ApkUIWorkload):

    platform = 'android'

    parameters = [
        Parameter('markers_enabled', kind=bool, default=False,
                  description="""
                  If set to ``True``, workloads will insert markers into logs
                  at various points during execution. These markes may be used
                  by other plugins or post-processing scripts to provide
                  measurments or statistics for specific parts of the workload
                  execution.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(ApkUiautoWorkload, self).__init__(target, **kwargs)
        self.gui = UiAutomatorGUI(self)

    def setup(self, context):
        self.gui.uiauto_params['package_name'] = self.apk.apk_info.package
        self.gui.uiauto_params['markers_enabled'] = self.markers_enabled
        self.gui.init_commands()
        super(ApkUiautoWorkload, self).setup(context)


class ApkReventWorkload(ApkUIWorkload):

    # May be optionally overwritten by subclasses
    # Times are in seconds
    setup_timeout = 5 * 60
    run_timeout = 10 * 60
    extract_results_timeout = 5 * 60
    teardown_timeout = 5 * 60

    def __init__(self, target, **kwargs):
        super(ApkReventWorkload, self).__init__(target, **kwargs)
        self.apk = PackageHandler(self)
        self.gui = ReventGUI(self, target,
                             self.setup_timeout,
                             self.run_timeout,
                             self.extract_results_timeout,
                             self.teardown_timeout)

class UIWorkload(Workload):

    def __init__(self, target, **kwargs):
        super(UIWorkload, self).__init__(target, **kwargs)
        self.gui = None

    def init_resources(self, context):
        super(UIWorkload, self).init_resources(context)
        self.gui.init_resources(context.resolver)

    @once_per_instance
    def initialize(self, context):
        super(UIWorkload, self).initialize(context)

    def setup(self, context):
        super(UIWorkload, self).setup(context)
        self.gui.deploy()
        self.gui.setup()

    def run(self, context):
        super(UIWorkload, self).run(context)
        self.gui.run()

    def extract_results(self, context):
        super(UIWorkload, self).extract_results(context)
        self.gui.extract_results()

    def teardown(self, context):
        self.gui.teardown()
        super(UIWorkload, self).teardown(context)

    @once_per_instance
    def finalize(self, context):
        super(UIWorkload, self).finalize(context)
        self.gui.remove()


class UiautoWorkload(UIWorkload):

    platform = 'android'

    parameters = [
        Parameter('markers_enabled', kind=bool, default=False,
                  description="""
                  If set to ``True``, workloads will insert markers into logs
                  at various points during execution. These markes may be used
                  by other plugins or post-processing scripts to provide
                  measurments or statistics for specific parts of the workload
                  execution.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(UiautoWorkload, self).__init__(target, **kwargs)
        self.gui = UiAutomatorGUI(self)

    def setup(self, context):
        self.gui.uiauto_params['markers_enabled'] = self.markers_enabled
        self.gui.init_commands()
        super(UiautoWorkload, self).setup(context)


class ReventWorkload(UIWorkload):

    # May be optionally overwritten by subclasses
    # Times are in seconds
    setup_timeout = 5 * 60
    run_timeout = 10 * 60
    extract_results_timeout = 5 * 60
    teardown_timeout = 5 * 60

    def __init__(self, target, **kwargs):
        super(ReventWorkload, self).__init__(target, **kwargs)
        self.gui = ReventGUI(self, target,
                             self.setup_timeout,
                             self.run_timeout,
                             self.extract_results_timeout,
                             self.teardown_timeout)



class UiAutomatorGUI(object):

    stages = ['setup', 'runWorkload', 'extractResults', 'teardown']

    uiauto_runner = 'android.support.test.runner.AndroidJUnitRunner'

    def __init__(self, owner, package=None, klass='UiAutomation', timeout=600):
        self.owner = owner
        self.target = self.owner.target
        self.uiauto_package = package
        self.uiauto_class = klass
        self.timeout = timeout
        self.logger = logging.getLogger('gui')
        self.uiauto_file = None
        self.commands = {}
        self.uiauto_params = ParameterDict()

    def init_resources(self, resolver):
        self.uiauto_file = resolver.get(ApkFile(self.owner, uiauto=True))
        if not self.uiauto_package:
            uiauto_info = ApkInfo(self.uiauto_file)
            self.uiauto_package = uiauto_info.package

    def init_commands(self):
        params_dict = self.uiauto_params
        params_dict['workdir'] = self.target.working_directory
        params = ''
        for k, v in params_dict.iter_encoded_items():
            params += ' -e {} {}'.format(k, v)

        for stage in self.stages:
            class_string = '{}.{}#{}'.format(self.uiauto_package, self.uiauto_class,
                                             stage)
            instrumentation_string = '{}/{}'.format(self.uiauto_package,
                                                    self.uiauto_runner)
            cmd_template = 'am instrument -w -r{} -e class {} {}'
            self.commands[stage] = cmd_template.format(params, class_string,
                                                       instrumentation_string)

    def deploy(self):
        if self.target.package_is_installed(self.uiauto_package):
            self.target.uninstall_package(self.uiauto_package)
        self.target.install_apk(self.uiauto_file)

    def set(self, name, value):
        self.uiauto_params[name] = value

    def setup(self, timeout=None):
        if not self.commands:
            raise RuntimeError('Commands have not been initialized')
        self.target.killall('uiautomator')
        self._execute('setup', timeout or self.timeout)

    def run(self, timeout=None):
        if not self.commands:
            raise RuntimeError('Commands have not been initialized')
        self._execute('runWorkload', timeout or self.timeout)

    def extract_results(self, timeout=None):
        if not self.commands:
            raise RuntimeError('Commands have not been initialized')
        self._execute('extractResults', timeout or self.timeout)

    def teardown(self, timeout=None):
        if not self.commands:
            raise RuntimeError('Commands have not been initialized')
        self._execute('teardown', timeout or self.timeout)

    def remove(self):
        self.target.uninstall(self.uiauto_package)

    def _execute(self, stage, timeout):
        result = self.target.execute(self.commands[stage], timeout)
        if 'FAILURE' in result:
            raise WorkloadError(result)
        else:
            self.logger.debug(result)
        time.sleep(2)


class ReventGUI(object):

    def __init__(self, workload, target, setup_timeout, run_timeout,
                 extract_results_timeout, teardown_timeout):
        self.workload = workload
        self.target = target
        self.setup_timeout = setup_timeout
        self.run_timeout = run_timeout
        self.extract_results_timeout = extract_results_timeout
        self.teardown_timeout = teardown_timeout
        self.revent_recorder = ReventRecorder(self.target)
        self.on_target_revent_binary = self.target.get_workpath('revent')
        self.on_target_setup_revent = self.target.get_workpath('{}.setup.revent'.format(self.target.model))
        self.on_target_run_revent = self.target.get_workpath('{}.run.revent'.format(self.target.model))
        self.on_target_extract_results_revent = self.target.get_workpath('{}.extract_results.revent'.format(self.target.model))
        self.on_target_teardown_revent = self.target.get_workpath('{}.teardown.revent'.format(self.target.model))
        self.logger = logging.getLogger('revent')
        self.revent_setup_file = None
        self.revent_run_file = None
        self.revent_extract_results_file = None
        self.revent_teardown_file = None

    def init_resources(self, resolver):
        self.revent_setup_file = resolver.get(ReventFile(owner=self.workload,
                                                         stage='setup',
                                                         target=self.target.model),
                                              strict=False)
        self.revent_run_file = resolver.get(ReventFile(owner=self.workload,
                                                       stage='run',
                                                       target=self.target.model))
        self.revent_extract_results_file = resolver.get(ReventFile(owner=self.workload,
                                                                 stage='extract_results',
                                                                 target=self.target.model),
                                                        strict=False)
        self.revent_teardown_file = resolver.get(resource=ReventFile(owner=self.workload,
                                                            stage='teardown',
                                                            target=self.target.model),
                                                 strict=False)

    def deploy(self):
        self.revent_recorder.deploy()

    def setup(self):
        self._check_revent_files()
        if self.revent_setup_file:
            self.revent_recorder.replay(self.on_target_setup_revent,
                                        timeout=self.setup_timeout)

    def run(self):
        msg = 'Replaying {}'
        self.logger.debug(msg.format(os.path.basename(self.on_target_run_revent)))
        self.revent_recorder.replay(self.on_target_run_revent,
                                    timeout=self.run_timeout)
        self.logger.debug('Replay completed.')

    def extract_results(self):
        if self.revent_extract_results_file:
            self.revent_recorder.replay(self.on_target_extract_results_revent,
                                        timeout=self.extract_results_timeout)

    def teardown(self):
        if self.revent_teardown_file:
            self.revent_recorder.replay(self.on_target_teardown_revent,
                                        timeout=self.teardown_timeout)
        self.target.remove(self.on_target_setup_revent)
        self.target.remove(self.on_target_run_revent)
        self.target.remove(self.on_target_extract_results_revent)
        self.target.remove(self.on_target_teardown_revent)

    def remove(self):
        self.revent_recorder.remove()

    def _check_revent_files(self):
        if not self.revent_run_file:
            # pylint: disable=too-few-format-args
            message = '{0}.run.revent file does not exist, ' \
                      'Please provide one for your target, {0}'
            raise WorkloadError(message.format(self.target.model))

        self.target.push(self.revent_run_file, self.on_target_run_revent)
        if self.revent_setup_file:
            self.target.push(self.revent_setup_file, self.on_target_setup_revent)
        if self.revent_extract_results_file:
            self.target.push(self.revent_extract_results_file, self.on_target_extract_results_revent)
        if self.revent_teardown_file:
            self.target.push(self.revent_teardown_file, self.on_target_teardown_revent)


class PackageHandler(object):

    @property
    def package(self):
        if self.apk_info is None:
            return None
        return self.apk_info.package

    @property
    def activity(self):
        if self.apk_info is None:
            return None
        return self.apk_info.activity

    def __init__(self, owner, install_timeout=300, version=None, variant=None,
                 package_name=None, strict=False, force_install=False, uninstall=False,
                 exact_abi=False):
        self.logger = logging.getLogger('apk')
        self.owner = owner
        self.target = self.owner.target
        self.install_timeout = install_timeout
        self.version = version
        self.variant = variant
        self.package_name = package_name
        self.strict = strict
        self.force_install = force_install
        self.uninstall = uninstall
        self.exact_abi = exact_abi
        self.apk_file = None
        self.apk_info = None
        self.apk_version = None
        self.logcat_log = None
        self.supported_abi = self.target.supported_abi

    def initialize(self, context):
        self.resolve_package(context)

    def setup(self, context):
        self.initialize_package(context)
        self.start_activity()
        self.target.execute('am kill-all')  # kill all *background* activities
        self.target.clear_logcat()

    def resolve_package(self, context):
        if not self.owner.package_names and not self.package_name:
            msg = 'Cannot Resolve package; No package name(s) specified'
            raise WorkloadError(msg)

        if self.package_name:
            self.apk_file = context.resolver.get(ApkFile(self.owner,
                                                         variant=self.variant,
                                                         version=self.version,
                                                         package=self.package_name,
                                                         exact_abi=self.exact_abi,
                                                         supported_abi=self.supported_abi),
                                                 strict=self.strict)
        else:
            available_packages = []
            for package in self.owner.package_names:
                available_packages.append(context.resolver.get(ApkFile(self.owner,
                                                                       variant=self.variant,
                                                                       version=self.version,
                                                                       package=package,
                                                                       exact_abi=self.exact_abi,
                                                                       supported_abi=self.supported_abi),
                                                               strict=self.strict))
            if len(available_packages) == 1:
                self.apk_file = available_packages[0]
            elif len(available_packages) > 1:
                msg = 'Multiple matching packages found for "{}" on host: {}'
                raise WorkloadError(msg.format(self.owner, available_packages))
        if self.apk_file:
            self.apk_info = ApkInfo(self.apk_file)
            if self.version:
                installed_version = self.target.get_package_version(self.apk_info.package)
                host_version = self.apk_info.version_name
                if (installed_version and installed_version != host_version and
                        loose_version_matching(self.version, installed_version)):
                    msg = 'Multiple matching packages found for {}; host version: {}, device version: {}'
                    raise WorkloadError(msg.format(self.owner, host_version, installed_version))
        else:
            self.resolve_package_from_target(context)

    def resolve_package_from_target(self, context):
        if self.package_name:
            if not self.target.package_is_installed(self.package_name):
                msg = 'Package "{}" cannot be found on the host or device'
                raise WorkloadError(msg.format(self.package_name))
        else:
            installed_versions = []
            for package in self.owner.package_names:
                if self.target.package_is_installed(package):
                    installed_versions.append(package)

            if self.version:
                for package in installed_versions:
                    package_version = self.target.get_package_version(package)
                    if loose_version_matching(self.version, package_version):
                        self.package_name = package
                        break
            else:
                if len(installed_versions) == 1:
                    self.package_name = installed_versions[0]
                elif len(installed_versions) > 1:
                    msg = 'Package version not set and multiple versions found on device'
                    raise WorkloadError(msg)

            if not self.package_name:
                raise WorkloadError('No matching package found')

        self.pull_apk(self.package_name)
        self.apk_file = context.resolver.get(ApkFile(self.owner,
                                                     variant=self.variant,
                                                     version=self.version,
                                                     package=self.package_name),
                                             strict=self.strict)
        self.apk_info = ApkInfo(self.apk_file)

    def initialize_package(self, context):
        installed_version = self.target.get_package_version(self.apk_info.package)
        host_version = self.apk_info.version_name
        if installed_version != host_version:
            if installed_version:
                message = '{} host version: {}, device version: {}; re-installing...'
                self.logger.debug(message.format(self.owner.name, host_version,
                                                 installed_version))
            else:
                message = '{} host version: {}, not found on device; installing...'
                self.logger.debug(message.format(self.owner.name, host_version))
            self.force_install = True  # pylint: disable=attribute-defined-outside-init
        else:
            message = '{} version {} found on both device and host.'
            self.logger.debug(message.format(self.owner.name, host_version))
        if self.force_install:
            if installed_version:
                self.target.uninstall_package(self.apk_info.package)
            self.install_apk(context)
        else:
            self.reset(context)
            if self.apk_info.permissions:
                self.logger.debug('Granting runtime permissions')
                for permission in self.apk_info.permissions:
                    self.target.grant_package_permission(self.apk_info.package, permission)
        self.apk_version = host_version

    def start_activity(self):
        if not self.apk_info.activity:
            cmd = 'am start -W {}'.format(self.apk_info.package)
        else:
            cmd = 'am start -W -n {}/{}'.format(self.apk_info.package,
                                                self.apk_info.activity)
        output = self.target.execute(cmd)
        if 'Error:' in output:
            # this will dismiss any error dialogs
            self.target.execute('am force-stop {}'.format(self.apk_info.package))
            raise WorkloadError(output)
        self.logger.debug(output)

    def reset(self, context):  # pylint: disable=W0613
        self.target.execute('am force-stop {}'.format(self.apk_info.package))
        self.target.execute('pm clear {}'.format(self.apk_info.package))

    def install_apk(self, context):
        # pylint: disable=unused-argument
        output = self.target.install_apk(self.apk_file, self.install_timeout,
                                         replace=True, allow_downgrade=True)
        if 'Failure' in output:
            if 'ALREADY_EXISTS' in output:
                msg = 'Using already installed APK (did not unistall properly?)'
                self.logger.warn(msg)
            else:
                raise WorkloadError(output)
        else:
            self.logger.debug(output)

    def pull_apk(self, package):
        if not self.target.package_is_installed(package):
            message = 'Cannot retrieve "{}" as not installed on Target'
            raise WorkloadError(message.format(package))
        package_info = self.target.execute('pm list packages -f {}'.format(package))
        apk_path = re.match('package:(.*)=', package_info).group(1)
        self.target.pull(apk_path, self.owner.dependencies_directory)

    def teardown(self):
        self.target.execute('am force-stop {}'.format(self.apk_info.package))
        if self.uninstall:
            self.target.uninstall_package(self.apk_info.package)
