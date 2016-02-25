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
import sys
import time

from wlauto.core.plugin import Parameter
from wlauto.core.workload import Workload
from wlauto.core.resource import NO_ONE
from wlauto.common.resources import PluginAsset, Executable
from wlauto.exceptions import WorkloadError, ResourceError, ConfigError
from wlauto.utils.android import ApkInfo, ANDROID_NORMAL_PERMISSIONS
from wlauto.utils.types import boolean
import wlauto.common.android.resources


DELAY = 5


class UiAutomatorWorkload(Workload):
    """
    Base class for all workloads that rely on a UI Automator JAR file.

    This class should be subclassed by workloads that rely on android UiAutomator
    to work. This class handles transferring the UI Automator JAR file to the device
    and invoking it to run the workload. By default, it will look for the JAR file in
    the same directory as the .py file for the workload (this can be changed by overriding
    the ``uiauto_file`` property in the subclassing workload).

    To inintiate UI Automation, the fully-qualified name of the Java class and the
    corresponding method name are needed. By default, the package part of the class name
    is derived from the class file, and class and method names are ``UiAutomation``
    and ``runUiAutomaton`` respectively. If you have generated the boilder plate for the
    UiAutomatior code using ``create_workloads`` utility, then everything should be named
    correctly. If you're creating the Java project manually, you need to make sure the names
    match what is expected, or you could override ``uiauto_package``, ``uiauto_class`` and
    ``uiauto_method`` class attributes with the value that match your Java code.

    You can also pass parameters to the JAR file. To do this add the parameters to
    ``self.uiauto_params`` dict inside your class's ``__init__`` or ``setup`` methods.

    """

    supported_platforms = ['android']

    uiauto_package = ''
    uiauto_class = 'UiAutomation'
    uiauto_method = 'runUiAutomation'

    # Can be overidden by subclasses to adjust to run time of specific
    # benchmarks.
    run_timeout = 4 * 60  # seconds

    def __init__(self, device, _call_super=True, **kwargs):  # pylint: disable=W0613
        if _call_super:
            super(UiAutomatorWorkload, self).__init__(device, **kwargs)
        self.uiauto_file = None
        self.device_uiauto_file = None
        self.command = None
        self.uiauto_params = {}

    def init_resources(self, context):
        self.uiauto_file = context.resolver.get(wlauto.common.android.resources.JarFile(self))
        if not self.uiauto_file:
            raise ResourceError('No UI automation JAR file found for workload {}.'.format(self.name))
        self.device_uiauto_file = self.device.path.join(self.device.working_directory,
                                                        os.path.basename(self.uiauto_file))
        if not self.uiauto_package:
            self.uiauto_package = os.path.splitext(os.path.basename(self.uiauto_file))[0]

    def setup(self, context):
        method_string = '{}.{}#{}'.format(self.uiauto_package, self.uiauto_class, self.uiauto_method)
        params_dict = self.uiauto_params
        params_dict['workdir'] = self.device.working_directory
        params = ''
        for k, v in self.uiauto_params.iteritems():
            params += ' -e {} {}'.format(k, v)
        self.command = 'uiautomator runtest {}{} -c {}'.format(self.device_uiauto_file, params, method_string)
        self.device.push(self.uiauto_file, self.device_uiauto_file)
        self.device.killall('uiautomator')

    def run(self, context):
        result = self.device.execute(self.command, self.run_timeout)
        if 'FAILURE' in result:
            raise WorkloadError(result)
        else:
            self.logger.debug(result)
        time.sleep(DELAY)

    def update_result(self, context):
        pass

    def teardown(self, context):
        self.device.remove(self.device_uiauto_file)

    def validate(self):
        if not self.uiauto_file:
            raise WorkloadError('No UI automation JAR file found for workload {}.'.format(self.name))
        if not self.uiauto_package:
            raise WorkloadError('No UI automation package specified for workload {}.'.format(self.name))


class ApkWorkload(Workload):
    """
    A workload based on an APK file.

    Defines the following attributes:

    :package: The package name of the app. This is usually a Java-style name of the form
              ``com.companyname.appname``.
    :activity: This is the initial activity of the app. This will be used to launch the
               app during the setup.
    :view: The class of the main view pane of the app. This needs to be defined in order
           to collect SurfaceFlinger-derived statistics (such as FPS) for the app, but
           may otherwise be left as ``None``.
    :install_timeout: Timeout for the installation of the APK. This may vary wildly based on
                      the size and nature of a specific APK, and so should be defined on
                      per-workload basis.

                      .. note:: To a lesser extent, this will also vary based on the the
                                device and the nature of adb connection (USB vs Ethernet),
                                so, as with all timeouts, so leeway must be included in
                                the specified value.

    .. note:: Both package and activity for a workload may be obtained from the APK using
              the ``aapt`` tool that comes with the ADT  (Android Developemnt Tools) bundle.

    """
    package = None
    activity = None
    view = None
    supported_platforms = ['android']

    parameters = [
        Parameter('install_timeout', kind=int, default=300,
                  description='Timeout for the installation of the apk.'),
        Parameter('check_apk', kind=boolean, default=True,
                  description='''
                  Discover the APK for this workload on the host, and check that
                  the version matches the one on device (if already installed).
                  '''),
        Parameter('force_install', kind=boolean, default=False,
                  description='''
                  Always re-install the APK, even if matching version is found
                  on already installed on the device.
                  '''),
        Parameter('uninstall_apk', kind=boolean, default=False,
                  description='If ``True``, will uninstall workload\'s APK as part of teardown.'),
    ]

    def __init__(self, device, _call_super=True, **kwargs):
        if _call_super:
            super(ApkWorkload, self).__init__(device, **kwargs)
        self.apk_file = None
        self.apk_version = None
        self.logcat_log = None

    def init_resources(self, context):
        self.apk_file = context.resolver.get(wlauto.common.android.resources.ApkFile(self),
                                             version=getattr(self, 'version', None),
                                             strict=self.check_apk)

    def validate(self):
        if self.check_apk:
            if not self.apk_file:
                raise WorkloadError('No APK file found for workload {}.'.format(self.name))
        else:
            if self.force_install:
                raise ConfigError('force_install cannot be "True" when check_apk is set to "False".')

    def setup(self, context):
        self.initialize_package(context)
        self.start_activity()
        self.device.execute('am kill-all')  # kill all *background* activities
        self.device.clear_logcat()

    def initialize_package(self, context):
        installed_version = self.device.get_package_version(self.package)
        if self.check_apk:
            self.initialize_with_host_apk(context, installed_version)
        else:
            if not installed_version:
                message = '''{} not found found on the device and check_apk is set to "False"
                             so host version was not checked.'''
                raise WorkloadError(message.format(self.package))
            message = 'Version {} installed on device; skipping host APK check.'
            self.logger.debug(message.format(installed_version))
            self.reset(context)
            self.apk_version = installed_version

    def initialize_with_host_apk(self, context, installed_version):
        host_version = ApkInfo(self.apk_file).version_name
        if installed_version != host_version:
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

        # As of android API level 23, apps can request permissions at runtime,
        # this will grant all of them so requests do not pop up when running the app
        if self.device.os_version['sdk'] >= 23:
            self._grant_requested_permissions()

    def install_apk(self, context):
        output = self.device.install(self.apk_file, self.install_timeout)
        if 'Failure' in output:
            if 'ALREADY_EXISTS' in output:
                self.logger.warn('Using already installed APK (did not unistall properly?)')
            else:
                raise WorkloadError(output)
        else:
            self.logger.debug(output)
        self.do_post_install(context)

    def _grant_requested_permissions(self):
        dumpsys_output = self.device.execute(command="dumpsys package {}".format(self.package))
        permissions = []
        lines = iter(dumpsys_output.splitlines())
        for line in lines:
            if "requested permissions:" in line:
                break

        for line in lines:
            if "android.permission." in line:
                permissions.append(line.split(":")[0].strip())
            else:
                break

        for permission in permissions:
            # "Normal" Permisions are automatically granted and cannot be changed
            permission_name = permission.rsplit('.', 1)[1]
            if permission_name not in ANDROID_NORMAL_PERMISSIONS:
                self.device.execute("pm grant {} {}".format(self.package, permission))

    def do_post_install(self, context):
        """ May be overwritten by dervied classes."""
        pass

    def run(self, context):
        pass

    def update_result(self, context):
        self.logcat_log = os.path.join(context.output_directory, 'logcat.log')
        context.device_manager.dump_logcat(self.logcat_log)
        context.add_iteration_artifact(name='logcat',
                                       path='logcat.log',
                                       kind='log',
                                       description='Logact dump for the run.')

    def teardown(self, context):
        self.device.execute('am force-stop {}'.format(self.package))
        if self.uninstall_apk:
            self.device.uninstall(self.package)


AndroidBenchmark = ApkWorkload  # backward compatibility


class ReventWorkload(Workload):

    default_setup_timeout = 5 * 60  # in seconds
    default_run_timeout = 10 * 60  # in seconds

    @property
    def on_device_setup_revent(self):
        return self.device.get_workpath('{}.setup.revent'.format(self.device.model))

    @property
    def on_device_run_revent(self):
        return self.device.get_workpath('{}.run.revent'.format(self.device.model))

    def __init__(self, device, _call_super=True, **kwargs):
        if _call_super:
            super(ReventWorkload, self).__init__(device, **kwargs)
        self.on_device_revent_binary = None
        self.setup_timeout = kwargs.get('setup_timeout', self.default_setup_timeout)
        self.run_timeout = kwargs.get('run_timeout', self.default_run_timeout)
        self.revent_setup_file = None
        self.revent_run_file = None

    def initialize(self, context):
        self.revent_setup_file = context.resolver.get(wlauto.common.android.resources.ReventFile(self, 'setup'))
        self.revent_run_file = context.resolver.get(wlauto.common.android.resources.ReventFile(self, 'run'))

    def setup(self, context):
        self._check_revent_files(context)
        self.device.killall('revent')
        command = '{} replay {}'.format(self.on_device_revent_binary, self.on_device_setup_revent)
        self.device.execute(command, timeout=self.setup_timeout)

    def run(self, context):
        command = '{} replay {}'.format(self.on_device_revent_binary, self.on_device_run_revent)
        self.logger.debug('Replaying {}'.format(os.path.basename(self.on_device_run_revent)))
        self.device.execute(command, timeout=self.run_timeout)
        self.logger.debug('Replay completed.')

    def update_result(self, context):
        pass

    def teardown(self, context):
        self.device.remove(self.on_device_setup_revent)
        self.device.remove(self.on_device_run_revent)

    def _check_revent_files(self, context):
        # check the revent binary
        revent_binary = context.resolver.get(Executable(NO_ONE, self.device.abi, 'revent'))
        if not os.path.isfile(revent_binary):
            message = '{} does not exist. '.format(revent_binary)
            message += 'Please build revent for your system and place it in that location'
            raise WorkloadError(message)
        if not self.revent_setup_file:
            # pylint: disable=too-few-format-args
            message = '{0}.setup.revent file does not exist, Please provide one for your device, {0}'.format(self.device.name)
            raise WorkloadError(message)
        if not self.revent_run_file:
            # pylint: disable=too-few-format-args
            message = '{0}.run.revent file does not exist, Please provide one for your device, {0}'.format(self.device.name)
            raise WorkloadError(message)

        self.on_device_revent_binary = self.device.install_executable(revent_binary)
        self.device.push(self.revent_run_file, self.on_device_run_revent)
        self.device.push(self.revent_setup_file, self.on_device_setup_revent)


class AndroidUiAutoBenchmark(UiAutomatorWorkload, AndroidBenchmark):

    supported_platforms = ['android']

    def __init__(self, device, **kwargs):
        UiAutomatorWorkload.__init__(self, device, **kwargs)
        AndroidBenchmark.__init__(self, device, _call_super=False, **kwargs)

    def init_resources(self, context):
        UiAutomatorWorkload.init_resources(self, context)
        AndroidBenchmark.init_resources(self, context)

    def setup(self, context):
        UiAutomatorWorkload.setup(self, context)
        AndroidBenchmark.setup(self, context)

    def update_result(self, context):
        UiAutomatorWorkload.update_result(self, context)
        AndroidBenchmark.update_result(self, context)

    def teardown(self, context):
        UiAutomatorWorkload.teardown(self, context)
        AndroidBenchmark.teardown(self, context)


class GameWorkload(ApkWorkload, ReventWorkload):
    """
    GameWorkload is the base class for all the workload that use revent files to
    run.

    For more in depth details on how to record revent files, please see
    :ref:`revent_files_creation`. To subclass this class, please refer to
    :ref:`GameWorkload`.

    Additionally, this class defines the following attributes:

    :asset_file: A tarball containing additional assets for the workload. These are the assets
                 that are not part of the APK but would need to be downloaded by the workload
                 (usually, on first run of the app). Since the presence of a network connection
                 cannot be assumed on some devices, this provides an alternative means of obtaining
                 the assets.
    :saved_state_file: A tarball containing the saved state for a workload. This tarball gets
                       deployed in the same way as the asset file. The only difference being that
                       it is usually much slower and re-deploying the tarball should alone be
                       enough to reset the workload to a known state (without having to reinstall
                       the app or re-deploy the other assets).
    :loading_time: Time it takes for the workload to load after the initial activity has been
                   started.

    """

    # May be optionally overwritten by subclasses
    asset_file = None
    saved_state_file = None
    view = 'SurfaceView'
    loading_time = 10
    supported_platforms = ['android']

    parameters = [
        Parameter('install_timeout', default=500, override=True),
        Parameter('assets_push_timeout', kind=int, default=500,
                  description='Timeout used during deployment of the assets package (if there is one).'),
        Parameter('clear_data_on_reset', kind=bool, default=True,
                  description="""
                  If set to ``False``, this will prevent WA from clearing package
                  data for this workload prior to running it.
                  """),
    ]

    def __init__(self, device, **kwargs):  # pylint: disable=W0613
        ApkWorkload.__init__(self, device, **kwargs)
        ReventWorkload.__init__(self, device, _call_super=False, **kwargs)
        self.logcat_process = None
        self.module_dir = os.path.dirname(sys.modules[self.__module__].__file__)
        self.revent_dir = os.path.join(self.module_dir, 'revent_files')

    def apk_init_resources(self, context):
        ApkWorkload.init_resources(self, context)

    def init_resources(self, context):
        self.apk_init_resources(context)
        ReventWorkload.init_resources(self, context)

    def setup(self, context):
        ApkWorkload.setup(self, context)
        self.logger.debug('Waiting for the game to load...')
        time.sleep(self.loading_time)
        ReventWorkload.setup(self, context)

    def do_post_install(self, context):
        ApkWorkload.do_post_install(self, context)
        self._deploy_assets(context, self.assets_push_timeout)

    def reset(self, context):
        # If saved state exists, restore it; if not, do full
        # uninstall/install cycle.
        self.device.execute('am force-stop {}'.format(self.package))
        if self.saved_state_file:
            self._deploy_resource_tarball(context, self.saved_state_file)
        else:
            if self.clear_data_on_reset:
                self.device.execute('pm clear {}'.format(self.package))
            self._deploy_assets(context)

    def run(self, context):
        ReventWorkload.run(self, context)

    def apk_teardown(self, context):
        if not self.saved_state_file:
            ApkWorkload.teardown(self, context)
        else:
            self.device.execute('am force-stop {}'.format(self.package))

    def teardown(self, context):
        self.apk_teardown(context)
        ReventWorkload.teardown(self, context)

    def _deploy_assets(self, context, timeout=300):
        if self.asset_file:
            self._deploy_resource_tarball(context, self.asset_file, timeout)
        if self.saved_state_file:  # must be deployed *after* asset tarball!
            self._deploy_resource_tarball(context, self.saved_state_file, timeout)

    def _deploy_resource_tarball(self, context, resource_file, timeout=300):
        kind = 'data'
        if ':' in resource_file:
            kind, resource_file = resource_file.split(':', 1)
        ondevice_cache = self.device.path.join(self.device.working_directory, '.cache', self.name, resource_file)
        if not self.device.file_exists(ondevice_cache):
            asset_tarball = context.resolver.get(PluginAsset(self, resource_file))
            if not asset_tarball:
                message = 'Could not find resource {} for workload {}.'
                raise WorkloadError(message.format(resource_file, self.name))
            # adb push will create intermediate directories if they don't
            # exist.
            self.device.push(asset_tarball, ondevice_cache, timeout=timeout)

        device_asset_directory = self.device.path.join(context.device_manager.external_storage_directory, 'Android', kind)
        deploy_command = 'cd {} && {} tar -xzf {}'.format(device_asset_directory,
                                                          self.device.busybox,
                                                          ondevice_cache)
        self.device.execute(deploy_command, timeout=timeout, as_root=True)
