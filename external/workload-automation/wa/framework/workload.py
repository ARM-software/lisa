#    Copyright 2014-2019 ARM Limited
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
import threading
import time

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from devlib.utils.android import ApkInfo

from wa.framework.plugin import TargetedPlugin, Parameter
from wa.framework.resource import (ApkFile, ReventFile,
                                   File, loose_version_matching,
                                   range_version_matching)
from wa.framework.exception import WorkloadError, ConfigError
from wa.utils.types import ParameterDict, list_or_string, version_tuple
from wa.utils.revent import ReventRecorder
from wa.utils.exec_control import once_per_instance


class Workload(TargetedPlugin):
    """
    This is the base class for the workloads executed by the framework.
    Each of the methods throwing NotImplementedError *must* be implemented
    by the derived classes.
    """

    kind = 'workload'

    parameters = [
        Parameter('cleanup_assets', kind=bool,
                  global_alias='cleanup_assets',
                  aliases=['clean_up'],
                  default=True,
                  description="""
                  If ``True``, assets that are deployed or created as part of the
                  workload will be removed again from the device.
                  """),
        Parameter('uninstall', kind=bool,
                  default=True,
                  description="""
                  If ``True``, executables that are installed to the device
                  as part of the workload will be uninstalled again.
                  """),
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

        supported_platforms = getattr(self, 'supported_platforms', [])
        if supported_platforms and self.target.os not in supported_platforms:
            msg = 'Supported platforms for "{}" are "{}", attempting to run on "{}"'
            raise WorkloadError(msg.format(self.name, ' '.join(self.supported_platforms),
                                           self.target.os))

    def init_resources(self, context):
        """
        This method may be used to perform early resource discovery and
        initialization. This is invoked during the initial loading stage and
        before the device is ready, so cannot be used for any device-dependent
        initialization. This method is invoked before the workload instance is
        validated.

        """
        for asset in self.deployable_assets:
            self.asset_files.append(context.get(File(self, asset)))

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
        metrics and artifacts for this workload iteration.

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

    supported_platforms = ['android']

    # May be optionally overwritten by subclasses
    # Times are in seconds
    loading_time = 10
    package_names = []
    supported_versions = []
    activity = None
    view = None
    clear_data_on_reset = True

    # Set this to True to mark that this workload requires the target apk to be run
    # for initialisation purposes before the main run is performed.
    requires_rerun = False

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
        Parameter('max_version', kind=str,
                  default=None,
                  description="""
                  The maximum version of the package to be used.
                  """),
        Parameter('min_version', kind=str,
                  default=None,
                  description="""
                  The minimum version of the package to be used.
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
                  override=True,
                  description="""
                  If ``True``, will uninstall workload\'s APK as part of teardown.'
                  """),
        Parameter('exact_abi', kind=bool,
                  default=False,
                  description="""
                  If ``True``, workload will check that the APK matches the target
                  device ABI, otherwise any suitable APK found will be used.
                  """),
        Parameter('prefer_host_package', kind=bool,
                  default=True,
                  aliases=['check_apk'],
                  description="""
                  If ``True`` then a package found on the host
                  will be preferred if it is a valid version and ABI, if not it
                  will fall back to the version on the target if available. If
                  ``False`` then the version on the target is preferred instead.
                  """),
        Parameter('view', kind=str, default=None, merge=True,
                  description="""
                  Manually override the 'View' of the workload for use with
                  instruments such as the ``fps`` instrument. If not specified,
                  a workload dependant 'View' will be automatically generated.
                  """),
    ]

    @property
    def package(self):
        return self.apk.package

    def __init__(self, target, **kwargs):
        if target.os == 'chromeos':
            if target.supports_android:
                target = target.android_container
            else:
                raise ConfigError('Target does not appear to support Android')

        super(ApkWorkload, self).__init__(target, **kwargs)

        if self.activity is not None and '.' not in self.activity:
            # If we're receiving just the activity name, it's taken relative to
            # the package namespace:
            self.activity = '.' + self.activity

        self.apk = PackageHandler(self,
                                  package_name=self.package_name,
                                  variant=self.variant,
                                  strict=self.strict,
                                  version=self.version or self.supported_versions,
                                  force_install=self.force_install,
                                  install_timeout=self.install_timeout,
                                  uninstall=self.uninstall,
                                  exact_abi=self.exact_abi,
                                  prefer_host_package=self.prefer_host_package,
                                  clear_data_on_reset=self.clear_data_on_reset,
                                  activity=self.activity,
                                  min_version=self.min_version,
                                  max_version=self.max_version)

    def validate(self):
        if self.min_version and self.max_version:
            if version_tuple(self.min_version) > version_tuple(self.max_version):
                msg = 'Cannot specify min version ({}) greater than max version ({})'
                raise ConfigError(msg.format(self.min_version, self.max_version))

    @once_per_instance
    def initialize(self, context):
        super(ApkWorkload, self).initialize(context)
        self.apk.initialize(context)
        # pylint: disable=access-member-before-definition, attribute-defined-outside-init
        if self.version is None:
            self.version = self.apk.apk_info.version_name
        if self.view is None:
            self.view = 'SurfaceView - {}/{}'.format(self.apk.package,
                                                     self.apk.activity)

    def setup(self, context):
        super(ApkWorkload, self).setup(context)
        self.apk.setup(context)
        if self.requires_rerun:
            self.setup_rerun()
            self.apk.restart_activity()
        time.sleep(self.loading_time)

    def setup_rerun(self):
        """
        Perform the setup necessary to rerun the workload. Only called if
        ``requires_rerun`` is set.
        """
        pass

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
        self.gui.init_resources(context)

    @once_per_instance
    def initialize(self, context):
        super(ApkUIWorkload, self).initialize(context)
        self.gui.deploy()

    def setup(self, context):
        super(ApkUIWorkload, self).setup(context)
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
        if self.cleanup_assets:
            self.gui.remove()


class ApkUiautoWorkload(ApkUIWorkload):

    parameters = [
        Parameter('markers_enabled', kind=bool, default=False,
                  description="""
                  If set to ``True``, workloads will insert markers into logs
                  at various points during execution. These markers may be used
                  by other plugins or post-processing scripts to provide
                  measurements or statistics for specific parts of the workload
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
        self.gui.init_resources(context)

    @once_per_instance
    def initialize(self, context):
        super(UIWorkload, self).initialize(context)
        self.gui.deploy()

    def setup(self, context):
        super(UIWorkload, self).setup(context)
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
        if self.cleanup_assets:
            self.gui.remove()


class UiautoWorkload(UIWorkload):

    supported_platforms = ['android']

    parameters = [
        Parameter('markers_enabled', kind=bool, default=False,
                  description="""
                  If set to ``True``, workloads will insert markers into logs
                  at various points during execution. These markers may be used
                  by other plugins or post-processing scripts to provide
                  measurements or statistics for specific parts of the workload
                  execution.
                  """),
    ]

    def __init__(self, target, **kwargs):
        if target.os == 'chromeos':
            if target.supports_android:
                target = target.android_container
            else:
                raise ConfigError('Target does not appear to support Android')

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

    def remove(self):
        self.target.remove(self.on_target_setup_revent)
        self.target.remove(self.on_target_run_revent)
        self.target.remove(self.on_target_extract_results_revent)
        self.target.remove(self.on_target_teardown_revent)
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
        if self._activity:
            return self._activity
        if self.apk_info is None:
            return None
        return self.apk_info.activity

    # pylint: disable=too-many-locals
    def __init__(self, owner, install_timeout=300, version=None, variant=None,
                 package_name=None, strict=False, force_install=False, uninstall=False,
                 exact_abi=False, prefer_host_package=True, clear_data_on_reset=True,
                 activity=None, min_version=None, max_version=None):
        self.logger = logging.getLogger('apk')
        self.owner = owner
        self.target = self.owner.target
        self.install_timeout = install_timeout
        self.version = version
        self.min_version = min_version
        self.max_version = max_version
        self.variant = variant
        self.package_name = package_name
        self.strict = strict
        self.force_install = force_install
        self.uninstall = uninstall
        self.exact_abi = exact_abi
        self.prefer_host_package = prefer_host_package
        self.clear_data_on_reset = clear_data_on_reset
        self._activity = activity
        self.supported_abi = self.target.supported_abi
        self.apk_file = None
        self.apk_info = None
        self.apk_version = None
        self.logcat_log = None
        self.error_msg = None

    def initialize(self, context):
        self.resolve_package(context)

    def setup(self, context):
        context.update_metadata('app_version', self.apk_info.version_name)
        self.initialize_package(context)
        self.start_activity()
        self.target.execute('am kill-all')  # kill all *background* activities
        self.target.clear_logcat()

    def resolve_package(self, context):
        if not self.owner.package_names and not self.package_name:
            msg = 'Cannot Resolve package; No package name(s) specified'
            raise WorkloadError(msg)

        self.error_msg = None
        if self.prefer_host_package:
            self.resolve_package_from_host(context)
            if not self.apk_file:
                self.resolve_package_from_target()
        else:
            self.resolve_package_from_target()
            if not self.apk_file:
                self.resolve_package_from_host(context)

        if self.apk_file:
            self.apk_info = ApkInfo(self.apk_file)
        else:
            if self.error_msg:
                raise WorkloadError(self.error_msg)
            else:
                if self.package_name:
                    message = 'Package "{package}" not found for workload {name} '\
                              'on host or target.'
                elif self.version:
                    message = 'No matching package found for workload {name} '\
                              '(version {version}) on host or target.'
                else:
                    message = 'No matching package found for workload {name} on host or target'
                raise WorkloadError(message.format(name=self.owner.name, version=self.version,
                                                   package=self.package_name))

    def resolve_package_from_host(self, context):
        self.logger.debug('Resolving package on host system')
        if self.package_name:
            self.apk_file = context.get_resource(ApkFile(self.owner,
                                                         variant=self.variant,
                                                         version=self.version,
                                                         package=self.package_name,
                                                         exact_abi=self.exact_abi,
                                                         supported_abi=self.supported_abi,
                                                         min_version=self.min_version,
                                                         max_version=self.max_version),
                                                 strict=self.strict)
        else:
            available_packages = []
            for package in self.owner.package_names:
                apk_file = context.get_resource(ApkFile(self.owner,
                                                        variant=self.variant,
                                                        version=self.version,
                                                        package=package,
                                                        exact_abi=self.exact_abi,
                                                        supported_abi=self.supported_abi,
                                                        min_version=self.min_version,
                                                        max_version=self.max_version),
                                                strict=self.strict)
                if apk_file:
                    available_packages.append(apk_file)
            if len(available_packages) == 1:
                self.apk_file = available_packages[0]
            elif len(available_packages) > 1:
                self.error_msg = self._get_package_error_msg('host')

    def resolve_package_from_target(self):  # pylint: disable=too-many-branches
        self.logger.debug('Resolving package on target')
        found_package = None
        if self.package_name:
            if not self.target.package_is_installed(self.package_name):
                return
            else:
                installed_versions = [self.package_name]
        else:
            installed_versions = []
            for package in self.owner.package_names:
                if self.target.package_is_installed(package):
                    installed_versions.append(package)

        if self.version or self.min_version or self.max_version:
            matching_packages = []
            for package in installed_versions:
                package_version = self.target.get_package_version(package)
                if self.version:
                    for v in list_or_string(self.version):
                        if loose_version_matching(v, package_version):
                            matching_packages.append(package)
                else:
                    if range_version_matching(package_version, self.min_version,
                                              self.max_version):
                        matching_packages.append(package)

            if len(matching_packages) == 1:
                found_package = matching_packages[0]
            elif len(matching_packages) > 1:
                self.error_msg = self._get_package_error_msg('device')
        else:
            if len(installed_versions) == 1:
                found_package = installed_versions[0]
            elif len(installed_versions) > 1:
                self.error_msg = 'Package version not set and multiple versions found on device.'
        if found_package:
            self.logger.debug('Found matching package on target; Pulling to host.')
            self.apk_file = self.pull_apk(found_package)
            self.package_name = found_package

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
            message = '{} version {} present on both device and host.'
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
        if not self.activity:
            cmd = 'am start -W {}'.format(self.apk_info.package)
        else:
            cmd = 'am start -W -n {}/{}'.format(self.apk_info.package,
                                                self.activity)
        output = self.target.execute(cmd)
        if 'Error:' in output:
            # this will dismiss any error dialogs
            self.target.execute('am force-stop {}'.format(self.apk_info.package))
            raise WorkloadError(output)
        self.logger.debug(output)

    def restart_activity(self):
        self.target.execute('am force-stop {}'.format(self.apk_info.package))
        self.start_activity()

    def reset(self, context):  # pylint: disable=W0613
        self.target.execute('am force-stop {}'.format(self.apk_info.package))
        if self.clear_data_on_reset:
            self.target.execute('pm clear {}'.format(self.apk_info.package))

    def install_apk(self, context):
        # pylint: disable=unused-argument
        output = self.target.install_apk(self.apk_file, self.install_timeout,
                                         replace=True, allow_downgrade=True)
        if 'Failure' in output:
            if 'ALREADY_EXISTS' in output:
                msg = 'Using already installed APK (did not uninstall properly?)'
                self.logger.warning(msg)
            else:
                raise WorkloadError(output)
        else:
            self.logger.debug(output)

    def pull_apk(self, package):
        if not self.target.package_is_installed(package):
            message = 'Cannot retrieve "{}" as not installed on Target'
            raise WorkloadError(message.format(package))
        package_info = self.target.get_package_info(package)
        self.target.pull(package_info.apk_path, self.owner.dependencies_directory,
                         timeout=self.install_timeout)
        apk_name = self.target.path.basename(package_info.apk_path)
        return os.path.join(self.owner.dependencies_directory, apk_name)

    def teardown(self):
        self.target.execute('am force-stop {}'.format(self.apk_info.package))
        if self.uninstall:
            self.target.uninstall_package(self.apk_info.package)

    def _get_package_error_msg(self, location):
        if self.version:
            msg = 'Multiple matches for "{version}" found on {location}.'
        elif self.min_version and self.max_version:
            msg = 'Multiple matches between versions "{min_version}" and "{max_version}" found on {location}.'
        elif self.max_version:
            msg = 'Multiple matches less than or equal to "{max_version}" found on {location}.'
        elif self.min_version:
            msg = 'Multiple matches greater or equal to "{min_version}" found on {location}.'
        else:
            msg = ''
        return msg.format(version=self.version, min_version=self.min_version,
                          max_version=self.max_version, location=location)


class TestPackageHandler(PackageHandler):
    """Class wrapping an APK used through ``am instrument``.
    """
    def __init__(self, owner, instrument_args=None, raw_output=False,
                 instrument_wait=True, no_hidden_api_checks=False,
                 *args, **kwargs):
        if instrument_args is None:
            instrument_args = {}
        super(TestPackageHandler, self).__init__(owner, *args, **kwargs)
        self.raw = raw_output
        self.args = instrument_args
        self.wait = instrument_wait
        self.no_checks = no_hidden_api_checks

        self.cmd = ''
        self.instrument_thread = None
        self._instrument_output = None

    def setup(self, context):
        self.initialize_package(context)

        words = ['am', 'instrument']
        if self.raw:
            words.append('-r')
        if self.wait:
            words.append('-w')
        if self.no_checks:
            words.append('--no-hidden-api-checks')
        for k, v in self.args.items():
            words.extend(['-e', str(k), str(v)])

        words.append(str(self.apk_info.package))
        if self.apk_info.activity:
            words[-1] += '/{}'.format(self.apk_info.activity)

        self.cmd = ' '.join(quote(x) for x in words)
        self.instrument_thread = threading.Thread(target=self._start_instrument)

    def start_activity(self):
        self.instrument_thread.start()

    def wait_instrument_over(self):
        self.instrument_thread.join()
        if 'Error:' in self._instrument_output:
            cmd = 'am force-stop {}'.format(self.apk_info.package)
            self.target.execute(cmd)
            raise WorkloadError(self._instrument_output)

    def _start_instrument(self):
        self._instrument_output = self.target.execute(self.cmd)
        self.logger.debug(self._instrument_output)

    @property
    def instrument_output(self):
        if self.instrument_thread.is_alive():
            self.instrument_thread.join()  # writes self._instrument_output
        return self._instrument_output
