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

# pylint: disable=no-member

"""
This module contains the execution logic for Workload Automation. It defines the
following actors:

    WorkloadSpec: Identifies the workload to be run and defines parameters under
                  which it should be executed.

    Executor: Responsible for the overall execution process. It instantiates
              and/or intialises the other actors, does any necessary vaidation
              and kicks off the whole process.

    Execution Context: Provides information about the current state of run
                       execution to instrumentation.

    RunInfo: Information about the current run.

    Runner: This executes workload specs that are passed to it. It goes through
            stages of execution, emitting an appropriate signal at each step to
            allow instrumentation to do its stuff.

"""
import logging
import os
import random
import subprocess
import uuid
from collections import Counter, defaultdict, OrderedDict
from contextlib import contextmanager
from copy import copy
from datetime import datetime
from itertools import izip_longest

import wa.framework.signal as signal
from wa.framework import instrumentation, pluginloader
from wa.framework.configuration.core import settings, RunStatus, JobStatus
from wa.framework.exception import (WAError, ConfigError, TimeoutError,
                                    InstrumentError, TargetError,
                                    TargetNotRespondingError)
from wa.framework.output import init_job_output
from wa.framework.plugin import Artifact
from wa.framework.resource import ResourceResolver
from wa.framework.run import RunState
from wa.framework.target.info import TargetInfo
from wa.framework.target.manager import TargetManager
from wa.utils import log
from wa.utils.misc import (ensure_directory_exists as _d, merge_config_values,
                           get_traceback, format_duration)
from wa.utils.serializer import json


# The maximum number of reboot attempts for an iteration.
MAX_REBOOT_ATTEMPTS = 3

# If something went wrong during device initialization, wait this
# long (in seconds) before retrying. This is necessary, as retrying
# immediately may not give the device enough time to recover to be able
# to reboot.
REBOOT_DELAY = 3


class ExecutionContext(object):

    @property
    def previous_job(self):
        if not self.job_queue:
            return None
        return self.job_queue[0]

    @property
    def next_job(self):
        if not self.completed_jobs:
            return None
        return self.completed_jobs[-1]

    @property
    def spec_changed(self):
        if self.previous_job is None and self.current_job is not None:  # Start of run
            return True
        if self.previous_job is not None and self.current_job is None:  # End of run
            return True
        return self.current_job.spec.id != self.previous_job.spec.id

    @property
    def spec_will_change(self):
        if self.current_job is None and self.next_job is not None:  # Start of run
            return True
        if self.current_job is not None and self.next_job is None:  # End of run
            return True
        return self.current_job.spec.id != self.next_job.spec.id

    @property
    def job_output(self):
        if self.current_job:
            return self.current_job.output

    @property
    def output(self):
        if self.current_job:
            return self.job_output
        return self.run_output

    @property
    def output_directory(self):
        return self.output.basepath

    def __init__(self, cm, tm, output):
        self.logger = logging.getLogger('context')
        self.cm = cm
        self.tm = tm
        self.run_output = output
        self.run_state = output.state
        self.logger.debug('Loading resource discoverers')
        self.resolver = ResourceResolver(cm)
        self.resolver.load()
        self.job_queue = None
        self.completed_jobs = None
        self.current_job = None

    def start_run(self):
        self.output.info.start_time = datetime.now()
        self.output.write_info()
        self.job_queue = copy(self.cm.jobs)
        self.completed_jobs = []
        self.run_state.status = RunStatus.STARTED
        self.output.write_state()

    def end_run(self):
        self.output.info.end_time = datetime.now()
        self.output.info.duration = self.output.info.end_time -\
                                    self.output.info.start_time
        self.output.write_info()
        self.output.write_state()
        self.output.write_result()

    def start_job(self):
        if not self.job_queue:
            raise RuntimeError('No jobs to run')
        self.current_job = self.job_queue.pop(0)
        self.current_job.output = init_job_output(self.run_output, self.current_job)
        self.update_job_state(self.current_job)
        return self.current_job

    def end_job(self):
        if not self.current_job:
            raise RuntimeError('No jobs in progress')
        self.completed_jobs.append(self.current_job)
        self.update_job_state(self.current_job)
        self.output.write_result()
        self.current_job = None

    def move_failed(self, job):
        self.run_output.move_failed(job.output)

    def update_job_state(self, job):
        self.run_state.update_job(job)
        self.run_output.write_state()

    def write_state(self):
        self.run_output.write_state()

    def add_metric(self, name, value, units=None, lower_is_better=False,
                   classifiers=None):
        if self.current_job:
            classifiers = merge_config_values(self.current_job.classifiers,
                                              classifiers)
        self.output.add_metric(name, value, units, lower_is_better, classifiers)

    def add_artifact(self, name, path, kind, description=None, classifiers=None):
        self.output.add_artifact(name, path, kind, description, classifiers)

    def add_run_artifact(self, name, path, kind, description=None,
                         classifiers=None):
        self.run_output.add_artifact(name, path, kind, description, classifiers)

class OldExecutionContext(object):
    """
    Provides a context for instrumentation. Keeps track of things like
    current workload and iteration.

    This class also provides two status members that can be used by workloads
    and instrumentation to keep track of arbitrary state. ``result``
    is reset on each new iteration of a workload; run_status is maintained
    throughout a Workload Automation run.

    """

    # These are the artifacts generated by the core framework.
    default_run_artifacts = [
        Artifact('runlog', 'run.log', 'log', mandatory=True,
                 description='The log for the entire run.'),
    ]

    @property
    def current_iteration(self):
        if self.current_job:
            spec_id = self.current_job.spec.id
            return self.job_iteration_counts[spec_id]
        else:
            return None

    @property
    def job_status(self):
        if not self.current_job:
            return None
        return self.current_job.result.status

    @property
    def workload(self):
        return getattr(self.spec, 'workload', None)

    @property
    def spec(self):
        return getattr(self.current_job, 'spec', None)

    @property
    def result(self):
        return getattr(self.current_job, 'result', self.run_result)

    def __init__(self, device_manager, config):
        self.device_manager = device_manager
        self.device = self.device_manager.target
        self.config = config
        self.reboot_policy = config.reboot_policy
        self.output_directory = None
        self.current_job = None
        self.resolver = None
        self.last_error = None
        self.run_info = None
        self.run_result = None
        self.run_output_directory = self.config.output_directory
        self.host_working_directory = self.config.meta_directory
        self.iteration_artifacts = None
        self.run_artifacts = copy(self.default_run_artifacts)
        self.job_iteration_counts = defaultdict(int)
        self.aborted = False
        self.runner = None

    def initialize(self):
        if not os.path.isdir(self.run_output_directory):
            os.makedirs(self.run_output_directory)
        self.output_directory = self.run_output_directory
        self.resolver = ResourceResolver(self.config)
        self.run_info = RunInfo(self.config)
        self.run_result = RunResult(self.run_info, self.run_output_directory)

    def next_job(self, job):
        """Invoked by the runner when starting a new iteration of workload execution."""
        self.current_job = job
        self.job_iteration_counts[self.spec.id] += 1
        if not self.aborted:
            outdir_name = '_'.join(map(str, [self.spec.label, self.spec.id, self.current_iteration]))
            self.output_directory = _d(os.path.join(self.run_output_directory, outdir_name))
            self.iteration_artifacts = [wa for wa in self.workload.artifacts]
        self.current_job.result.iteration = self.current_iteration
        self.current_job.result.output_directory = self.output_directory

    def end_job(self):
        if self.current_job.result.status == JobStatus.ABORTED:
            self.aborted = True
        self.current_job = None
        self.output_directory = self.run_output_directory

    def add_metric(self, *args, **kwargs):
        self.result.add_metric(*args, **kwargs)

    def add_artifact(self, name, path, kind, *args, **kwargs):
        if self.current_job is None:
            self.add_run_artifact(name, path, kind, *args, **kwargs)
        else:
            self.add_iteration_artifact(name, path, kind, *args, **kwargs)

    def add_run_artifact(self, name, path, kind, *args, **kwargs):
        path = _check_artifact_path(path, self.run_output_directory)
        self.run_artifacts.append(Artifact(name, path, kind, Artifact.ITERATION, *args, **kwargs))

    def add_iteration_artifact(self, name, path, kind, *args, **kwargs):
        path = _check_artifact_path(path, self.output_directory)
        self.iteration_artifacts.append(Artifact(name, path, kind, Artifact.RUN, *args, **kwargs))

    def get_artifact(self, name):
        if self.iteration_artifacts:
            for art in self.iteration_artifacts:
                if art.name == name:
                    return art
        for art in self.run_artifacts:
            if art.name == name:
                return art
        return None


def _check_artifact_path(path, rootpath):
    if path.startswith(rootpath):
        return os.path.abspath(path)
    rootpath = os.path.abspath(rootpath)
    full_path = os.path.join(rootpath, path)
    if not os.path.isfile(full_path):
        msg = 'Cannot add artifact because {} does not exist.'
        raise ValueError(msg.format(full_path))
    return full_path


class Executor(object):
    """
    The ``Executor``'s job is to set up the execution context and pass to a
    ``Runner`` along with a loaded run specification. Once the ``Runner`` has
    done its thing, the ``Executor`` performs some final reporint before
    returning.

    The initial context set up involves combining configuration from various
    sources, loading of requided workloads, loading and installation of
    instruments and result processors, etc. Static validation of the combined
    configuration is also performed.

    """
    # pylint: disable=R0915

    def __init__(self):
        self.logger = logging.getLogger('executor')
        self.error_logged = False
        self.warning_logged = False
        pluginloader = None
        self.device_manager = None
        self.device = None
        self.context = None

    def execute(self, config_manager, output):
        """
        Execute the run specified by an agenda. Optionally, selectors may be
        used to only selecute a subset of the specified agenda.

        Params::

            :state: a ``ConfigManager`` containing processed configuraiton
            :output: an initialized ``RunOutput`` that will be used to
                     store the results.

        """
        signal.connect(self._error_signalled_callback, signal.ERROR_LOGGED)
        signal.connect(self._warning_signalled_callback, signal.WARNING_LOGGED)

        self.logger.info('Initializing run')
        self.logger.debug('Finalizing run configuration.')
        config = config_manager.finalize()
        output.write_config(config)

        self.logger.info('Connecting to target')
        target_manager = TargetManager(config.run_config.device,
                                       config.run_config.device_config)
        output.write_target_info(target_manager.get_target_info())

        self.logger.info('Initializing execution conetext')
        context = ExecutionContext(config_manager, target_manager, output)

        self.logger.info('Generating jobs')
        config_manager.generate_jobs(context)
        output.write_job_specs(config_manager.job_specs)
        output.write_state()

        self.logger.info('Installing instrumentation')
        for instrument in config_manager.get_instruments(target_manager.target):
            instrumentation.install(instrument)
        instrumentation.validate()

        self.logger.info('Starting run')
        runner = Runner(context)
        runner.run()

    def execute_postamble(self):
        """
        This happens after the run has completed. The overall results of the run are
        summarised to the user.

        """
        result = self.context.run_result
        counter = Counter()
        for ir in result.iteration_results:
            counter[ir.status] += 1
        self.logger.info('Done.')
        self.logger.info('Run duration: {}'.format(format_duration(self.context.run_info.duration)))
        status_summary = 'Ran a total of {} iterations: '.format(sum(self.context.job_iteration_counts.values()))
        parts = []
        for status in JobStatus.values:
            if status in counter:
                parts.append('{} {}'.format(counter[status], status))
        self.logger.info(status_summary + ', '.join(parts))
        self.logger.info('Results can be found in {}'.format(self.config.output_directory))

        if self.error_logged:
            self.logger.warn('There were errors during execution.')
            self.logger.warn('Please see {}'.format(self.config.log_file))
        elif self.warning_logged:
            self.logger.warn('There were warnings during execution.')
            self.logger.warn('Please see {}'.format(self.config.log_file))

    def _error_signalled_callback(self):
        self.error_logged = True
        signal.disconnect(self._error_signalled_callback, signal.ERROR_LOGGED)

    def _warning_signalled_callback(self):
        self.warning_logged = True
        signal.disconnect(self._warning_signalled_callback, signal.WARNING_LOGGED)


class Runner(object):
    """
    
    """

    def __init__(self, context):
        self.logger = logging.getLogger('runner')
        self.context = context
        self.output = self.context.output
        self.config = self.context.cm

    def run(self):
        self.send(signal.RUN_STARTED)
        try:
            self.initialize_run()
            self.send(signal.RUN_INITIALIZED)

            while self.context.job_queue:
                with signal.wrap('JOB_EXECUTION', self):
                    self.run_next_job(self.context)
        except Exception as e:
            if (not getattr(e, 'logged', None) and
                    not isinstance(e, KeyboardInterrupt)):
                log.log_error(e, self.logger)
                e.logged = True
            raise e
        finally:
            self.finalize_run()
            self.send(signal.RUN_COMPLETED)

    def initialize_run(self):
        self.logger.info('Initializing run')
        self.context.start_run()
        log.indent()
        for job in self.context.job_queue:
            job.initialize(self.context)
        log.dedent()
        self.context.write_state()

    def finalize_run(self):
        self.logger.info('Finalizing run')
        self.context.end_run()

    def run_next_job(self, context):
        job = context.start_job()
        self.logger.info('Running job {}'.format(job.id))

        try:
            log.indent()
            self.do_run_job(job, context)
            job.status = JobStatus.OK
        except KeyboardInterrupt:
            job.status = JobStatus.ABORTED
            raise
        except Exception as e:
            job.status = JobStatus.FAILED
            if not getattr(e, 'logged', None):
                log.log_error(e, self.logger)
                e.logged = True
        finally:
            self.logger.info('Completing job {}'.format(job.id))
            self.send(signal.JOB_COMPLETED)
            context.end_job()

            log.dedent()
            self.check_job(job)

    def do_run_job(self, job, context):
        job.status = JobStatus.RUNNING
        self.send(signal.JOB_STARTED)

        with signal.wrap('JOB_TARGET_CONFIG', self):
            job.configure_target(context)

        with signal.wrap('JOB_SETUP', self):
            job.setup(context)
        
        try:
            with signal.wrap('JOB_EXECUTION', self):
                job.run(context)

            try:
                with signal.wrap('JOB_OUTPUT_PROCESSED', self):
                    job.process_output(context)
            except Exception:
                job.status = JobStatus.PARTIAL
                raise
        except KeyboardInterrupt:
            job.status = JobStatus.ABORTED
            raise
        except Exception as e:
            job.status = JobStatus.FAILED
            if not getattr(e, 'logged', None):
                log.log_error(e, self.logger)
                e.logged = True
            raise e
        finally:
            # If setup was successfully completed, teardown must
            # run even if the job failed
            with signal.wrap('JOB_TEARDOWN', self):
                job.teardown(context)

    def check_job(self, job):
        rc = self.context.cm.run_config
        if job.status in rc.retry_on_status:
            if job.retries < rc.max_retries:
                msg = 'Job {} iteration {} completed with status {}. retrying...'
                self.logger.error(msg.format(job.id, job.status, job.iteration))
                self.context.move_failed(job)
                job.retries += 1
                job.status = JobStatus.PENDING
                self.context.job_queue.insert(0, job)
                self.context.write_state()
            else:
                msg = 'Job {} iteration {} completed with status {}. '\
                      'Max retries exceeded.'
                self.logger.error(msg.format(job.id, job.status, job.iteration))
        else:  # status not in retry_on_status
            self.logger.info('Job completed with status {}'.format(job.status))
        
    def send(self, s):
        signal.send(s, self, self.context)

    def __str__(self):
        return 'runner'


class RunnerJob(object):
    """
    Represents a single execution of a ``RunnerJobDescription``. There will be one created for each iteration
    specified by ``RunnerJobDescription.number_of_iterations``.

    """

    def __init__(self, spec, retry=0):
        self.spec = spec
        self.retry = retry
        self.iteration = None
        self.result = JobStatus(self.spec)


class OldRunner(object):
    """
    This class is responsible for actually performing a workload automation
    run. The main responsibility of this class is to emit appropriate signals
    at the various stages of the run to allow things like traces an other
    instrumentation to hook into the process.

    This is an abstract base class that defines each step of the run, but not
    the order in which those steps are executed, which is left to the concrete
    derived classes.

    """
    class _RunnerError(Exception):
        """Internal runner error."""
        pass

    @property
    def config(self):
        return self.context.config

    @property
    def current_job(self):
        if self.job_queue:
            return self.job_queue[0]
        return None

    @property
    def previous_job(self):
        if self.completed_jobs:
            return self.completed_jobs[-1]
        return None

    @property
    def next_job(self):
        if self.job_queue:
            if len(self.job_queue) > 1:
                return self.job_queue[1]
        return None

    @property
    def spec_changed(self):
        if self.previous_job is None and self.current_job is not None:  # Start of run
            return True
        if self.previous_job is not None and self.current_job is None:  # End of run
            return True
        return self.current_job.spec.id != self.previous_job.spec.id

    @property
    def spec_will_change(self):
        if self.current_job is None and self.next_job is not None:  # Start of run
            return True
        if self.current_job is not None and self.next_job is None:  # End of run
            return True
        return self.current_job.spec.id != self.next_job.spec.id

    def __init__(self, device_manager, context, result_manager):
        self.device_manager = device_manager
        self.device = device_manager.target
        self.context = context
        self.result_manager = result_manager
        self.logger = logging.getLogger('Runner')
        self.job_queue = []
        self.completed_jobs = []
        self._initial_reset = True

    def init_queue(self, specs):
        raise NotImplementedError()

    def run(self):  # pylint: disable=too-many-branches
        self._send(signal.RUN_START)
        with signal.wrap('RUN_INIT'):
            self._initialize_run()

        try:
            while self.job_queue:
                try:
                    self._init_job()
                    self._run_job()
                except KeyboardInterrupt:
                    self.current_job.result.status = JobStatus.ABORTED
                    raise
                except Exception, e:  # pylint: disable=broad-except
                    self.current_job.result.status = JobStatus.FAILED
                    self.current_job.result.add_event(e.message)
                    if isinstance(e, DeviceNotRespondingError):
                        self.logger.info('Device appears to be unresponsive.')
                        if self.context.reboot_policy.can_reboot and self.device.can('reset_power'):
                            self.logger.info('Attempting to hard-reset the device...')
                            try:
                                self.device.boot(hard=True)
                                self.device.connect()
                            except DeviceError:  # hard_boot not implemented for the device.
                                raise e
                        else:
                            raise e
                    else:  # not a DeviceNotRespondingError
                        self.logger.error(e)
                finally:
                    self._finalize_job()
        except KeyboardInterrupt:
            self.logger.info('Got CTRL-C. Finalizing run... (CTRL-C again to abort).')
            # Skip through the remaining jobs.
            while self.job_queue:
                self.context.next_job(self.current_job)
                self.current_job.result.status = JobStatus.ABORTED
                self._finalize_job()
        except DeviceNotRespondingError:
            self.logger.info('Device unresponsive and recovery not possible. Skipping the rest of the run.')
            self.context.aborted = True
            while self.job_queue:
                self.context.next_job(self.current_job)
                self.current_job.result.status = JobStatus.SKIPPED
                self._finalize_job()

        instrumentation.enable_all()
        self._finalize_run()
        self._process_results()

        self.result_manager.finalize(self.context)
        self._send(signal.RUN_END)

    def _initialize_run(self):
        self.context.runner = self
        self.context.run_info.start_time = datetime.utcnow()
        self._connect_to_device()
        self.logger.info('Initializing device')
        self.device_manager.initialize(self.context)

        self.logger.info('Initializing workloads')
        for workload_spec in self.context.config.workload_specs:
            workload_spec.workload.initialize(self.context)

        self.context.run_info.device_properties = self.device_manager.info
        self.result_manager.initialize(self.context)

        if instrumentation.check_failures():
            raise InstrumentError('Detected failure(s) during instrumentation initialization.')

    def _connect_to_device(self):
        if self.context.reboot_policy.perform_initial_boot:
            try:
                self.device_manager.connect()
            except DeviceError:  # device may be offline
                if self.device.can('reset_power'):
                    with self._signal_wrap('INITIAL_BOOT'):
                        self.device.boot(hard=True)
                else:
                    raise DeviceError('Cannot connect to device for initial reboot; '
                                      'and device does not support hard reset.')
            else:  # successfully connected
                self.logger.info('\tBooting device')
                with self._signal_wrap('INITIAL_BOOT'):
                    self._reboot_device()
        else:
            self.logger.info('Connecting to device')
            self.device_manager.connect()

    def _init_job(self):
        self.current_job.result.status = JobStatus.RUNNING
        self.context.next_job(self.current_job)

    def _run_job(self):   # pylint: disable=too-many-branches
        spec = self.current_job.spec
        if not spec.enabled:
            self.logger.info('Skipping workload %s (iteration %s)', spec, self.context.current_iteration)
            self.current_job.result.status = JobStatus.SKIPPED
            return

        self.logger.info('Running workload %s (iteration %s)', spec, self.context.current_iteration)
        if spec.flash:
            if not self.context.reboot_policy.can_reboot:
                raise ConfigError('Cannot flash as reboot_policy does not permit rebooting.')
            if not self.device.can('flash'):
                raise DeviceError('Device does not support flashing.')
            self._flash_device(spec.flash)
        elif not self.completed_jobs:
            # Never reboot on the very fist job of a run, as we would have done
            # the initial reboot if a reboot was needed.
            pass
        elif self.context.reboot_policy.reboot_on_each_spec and self.spec_changed:
            self.logger.debug('Rebooting on spec change.')
            self._reboot_device()
        elif self.context.reboot_policy.reboot_on_each_iteration:
            self.logger.debug('Rebooting on iteration.')
            self._reboot_device()

        instrumentation.disable_all()
        instrumentation.enable(spec.instrumentation)
        self.device_manager.start()

        if self.spec_changed:
            self._send(signal.WORKLOAD_SPEC_START)
        self._send(signal.ITERATION_START)

        try:
            setup_ok = False
            with self._handle_errors('Setting up device parameters'):
                self.device_manager.set_runtime_parameters(spec.runtime_parameters)
                setup_ok = True

            if setup_ok:
                with self._handle_errors('running {}'.format(spec.workload.name)):
                    self.current_job.result.status = JobStatus.RUNNING
                    self._run_workload_iteration(spec.workload)
            else:
                self.logger.info('\tSkipping the rest of the iterations for this spec.')
                spec.enabled = False
        except KeyboardInterrupt:
            self._send(signal.ITERATION_END)
            self._send(signal.WORKLOAD_SPEC_END)
            raise
        else:
            self._send(signal.ITERATION_END)
            if self.spec_will_change or not spec.enabled:
                self._send(signal.WORKLOAD_SPEC_END)
        finally:
            self.device_manager.stop()

    def _finalize_job(self):
        self.context.run_result.iteration_results.append(self.current_job.result)
        job = self.job_queue.pop(0)
        job.iteration = self.context.current_iteration
        if job.result.status in self.config.retry_on_status:
            if job.retry >= self.config.max_retries:
                self.logger.error('Exceeded maxium number of retries. Abandoning job.')
            else:
                self.logger.info('Job status was {}. Retrying...'.format(job.result.status))
                retry_job = RunnerJob(job.spec, job.retry + 1)
                self.job_queue.insert(0, retry_job)
        self.completed_jobs.append(job)
        self.context.end_job()

    def _finalize_run(self):
        self.logger.info('Finalizing workloads')
        for workload_spec in self.context.config.workload_specs:
            workload_spec.workload.finalize(self.context)

        self.logger.info('Finalizing.')
        self._send(signal.RUN_FIN)

        with self._handle_errors('Disconnecting from the device'):
            self.device.disconnect()

        info = self.context.run_info
        info.end_time = datetime.utcnow()
        info.duration = info.end_time - info.start_time

    def _process_results(self):
        self.logger.info('Processing overall results')
        with self._signal_wrap('OVERALL_RESULTS_PROCESSING'):
            if instrumentation.check_failures():
                self.context.run_result.non_iteration_errors = True
            self.result_manager.process_run_result(self.context.run_result, self.context)

    def _run_workload_iteration(self, workload):
        self.logger.info('\tSetting up')
        with self._signal_wrap('WORKLOAD_SETUP'):
            try:
                workload.setup(self.context)
            except:
                self.logger.info('\tSkipping the rest of the iterations for this spec.')
                self.current_job.spec.enabled = False
                raise
        try:

            self.logger.info('\tExecuting')
            with self._handle_errors('Running workload'):
                with self._signal_wrap('WORKLOAD_EXECUTION'):
                    workload.run(self.context)

            self.logger.info('\tProcessing result')
            self._send(signal.BEFORE_WORKLOAD_RESULT_UPDATE)
            try:
                if self.current_job.result.status != JobStatus.FAILED:
                    with self._handle_errors('Processing workload result',
                                             on_error_status=JobStatus.PARTIAL):
                        workload.update_result(self.context)
                        self._send(signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE)

                if self.current_job.result.status == JobStatus.RUNNING:
                    self.current_job.result.status = JobStatus.OK
            finally:
                self._send(signal.AFTER_WORKLOAD_RESULT_UPDATE)

        finally:
            self.logger.info('\tTearing down')
            with self._handle_errors('Tearing down workload',
                                     on_error_status=JobStatus.NONCRITICAL):
                with self._signal_wrap('WORKLOAD_TEARDOWN'):
                    workload.teardown(self.context)
            self.result_manager.add_result(self.current_job.result, self.context)

    def _flash_device(self, flashing_params):
        with self._signal_wrap('FLASHING'):
            self.device.flash(**flashing_params)
            self.device.connect()

    def _reboot_device(self):
        with self._signal_wrap('BOOT'):
            for reboot_attempts in xrange(MAX_REBOOT_ATTEMPTS):
                if reboot_attempts:
                    self.logger.info('\tRetrying...')
                with self._handle_errors('Rebooting device'):
                    self.device.boot(**self.current_job.spec.boot_parameters)
                    break
            else:
                raise DeviceError('Could not reboot device; max reboot attempts exceeded.')
            self.device.connect()

    def _send(self, s):
        signal.send(s, self, self.context)

    def _take_screenshot(self, filename):
        if self.context.output_directory:
            filepath = os.path.join(self.context.output_directory, filename)
        else:
            filepath = os.path.join(settings.output_directory, filename)
        self.device.capture_screen(filepath)

    @contextmanager
    def _handle_errors(self, action, on_error_status=JobStatus.FAILED):
        try:
            if action is not None:
                self.logger.debug(action)
            yield
        except (KeyboardInterrupt, DeviceNotRespondingError):
            raise
        except (WAError, TimeoutError), we:
            self.device.check_responsive()
            if self.current_job:
                self.current_job.result.status = on_error_status
                self.current_job.result.add_event(str(we))
            try:
                self._take_screenshot('error.png')
            except Exception, e:  # pylint: disable=W0703
                # We're already in error state, so the fact that taking a
                # screenshot failed is not surprising...
                pass
            if action:
                action = action[0].lower() + action[1:]
            self.logger.error('Error while {}:\n\t{}'.format(action, we))
        except Exception, e:  # pylint: disable=W0703
            error_text = '{}("{}")'.format(e.__class__.__name__, e)
            if self.current_job:
                self.current_job.result.status = on_error_status
                self.current_job.result.add_event(error_text)
            self.logger.error('Error while {}'.format(action))
            self.logger.error(error_text)
            if isinstance(e, subprocess.CalledProcessError):
                self.logger.error('Got:')
                self.logger.error(e.output)
            tb = get_traceback()
            self.logger.error(tb)

    @contextmanager
    def _signal_wrap(self, signal_name):
        """Wraps the suite in before/after signals, ensuring
        that after signal is always sent."""
        before_signal = getattr(signal, 'BEFORE_' + signal_name)
        success_signal = getattr(signal, 'SUCCESSFUL_' + signal_name)
        after_signal = getattr(signal, 'AFTER_' + signal_name)
        try:
            self._send(before_signal)
            yield
            self._send(success_signal)
        finally:
            self._send(after_signal)


class BySpecRunner(Runner):
    """
    This is that "classic" implementation that executes all iterations of a workload
    spec before proceeding onto the next spec.

    """

    def init_queue(self, specs):
        jobs = [[RunnerJob(s) for _ in xrange(s.number_of_iterations)] for s in specs]  # pylint: disable=unused-variable
        self.job_queue = [j for spec_jobs in jobs for j in spec_jobs]


class BySectionRunner(Runner):
    """
    Runs the first iteration for all benchmarks first, before proceeding to the next iteration,
    i.e. A1, B1, C1, A2, B2, C2...  instead of  A1, A1, B1, B2, C1, C2...

    If multiple sections where specified in the agenda, this will run all specs for the first section
    followed by all specs for the seciod section, etc.

    e.g. given sections X and Y, and global specs A and B, with 2 iterations, this will run

    X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2

    """

    def init_queue(self, specs):
        jobs = [[RunnerJob(s) for _ in xrange(s.number_of_iterations)] for s in specs]
        self.job_queue = [j for spec_jobs in izip_longest(*jobs) for j in spec_jobs if j]


class ByIterationRunner(Runner):
    """
    Runs the first iteration for all benchmarks first, before proceeding to the next iteration,
    i.e. A1, B1, C1, A2, B2, C2...  instead of  A1, A1, B1, B2, C1, C2...

    If multiple sections where specified in the agenda, this will run all sections for the first global
    spec first, followed by all sections for the second spec, etc.

    e.g. given sections X and Y, and global specs A and B, with 2 iterations, this will run

    X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2

    """

    def init_queue(self, specs):
        sections = OrderedDict()
        for s in specs:
            if s.section_id not in sections:
                sections[s.section_id] = []
            sections[s.section_id].append(s)
        specs = [s for section_specs in izip_longest(*sections.values()) for s in section_specs if s]
        jobs = [[RunnerJob(s) for _ in xrange(s.number_of_iterations)] for s in specs]
        self.job_queue = [j for spec_jobs in izip_longest(*jobs) for j in spec_jobs if j]


class RandomRunner(Runner):
    """
    This will run specs in a random order.

    """

    def init_queue(self, specs):
        jobs = [[RunnerJob(s) for _ in xrange(s.number_of_iterations)] for s in specs]  # pylint: disable=unused-variable
        all_jobs = [j for spec_jobs in jobs for j in spec_jobs]
        random.shuffle(all_jobs)
        self.job_queue = all_jobs
