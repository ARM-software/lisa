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

import logging
from copy import copy
from datetime import datetime

import wa.framework.signal as signal
from wa.framework import instrumentation
from wa.framework.configuration.core import Status
from wa.framework.exception import HostError, WorkloadError
from wa.framework.job import Job
from wa.framework.output import init_job_output
from wa.framework.processor import ProcessorManager
from wa.framework.resource import ResourceResolver
from wa.framework.target.manager import TargetManager
from wa.utils import log
from wa.utils.misc import merge_config_values, format_duration


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
    def workload(self):
        if self.current_job:
            return self.current_job.workload

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
        self.target_info = self.tm.get_target_info()
        self.logger.debug('Loading resource discoverers')
        self.resolver = ResourceResolver(cm.plugin_cache)
        self.resolver.load()
        self.job_queue = None
        self.completed_jobs = None
        self.current_job = None
        self.successful_jobs = 0
        self.failed_jobs = 0

    def start_run(self):
        self.output.info.start_time = datetime.utcnow()
        self.output.write_info()
        self.job_queue = copy(self.cm.jobs)
        self.completed_jobs = []
        self.run_state.status = Status.STARTED
        self.output.status = Status.STARTED
        self.output.write_state()

    def end_run(self):
        if self.successful_jobs:
            if self.failed_jobs:
                status = Status.PARTIAL
            else:
                status = Status.OK
        else:
            status = Status.FAILED
        self.run_state.status = status
        self.output.status = status
        self.output.info.end_time = datetime.utcnow()
        self.output.info.duration = self.output.info.end_time -\
                                    self.output.info.start_time
        self.output.write_info()
        self.output.write_state()
        self.output.write_result()

    def finalize(self):
        self.tm.finalize()

    def start_job(self):
        if not self.job_queue:
            raise RuntimeError('No jobs to run')
        self.current_job = self.job_queue.pop(0)
        self.current_job.output = init_job_output(self.run_output, self.current_job)
        self.update_job_state(self.current_job)
        self.tm.start()
        return self.current_job

    def end_job(self):
        if not self.current_job:
            raise RuntimeError('No jobs in progress')
        self.tm.stop()
        self.completed_jobs.append(self.current_job)
        self.update_job_state(self.current_job)
        self.output.write_result()
        self.current_job = None

    def set_status(self, status, force=False):
        if not self.current_job:
            raise RuntimeError('No jobs in progress')
        self.current_job.set_status(status, force)

    def extract_results(self):
        self.tm.extract_results(self)

    def move_failed(self, job):
        self.run_output.move_failed(job.output)

    def update_job_state(self, job):
        self.run_state.update_job(job)
        self.run_output.write_state()

    def skip_job(self, job):
        job.status = Status.SKIPPED
        self.run_state.update_job(job)
        self.completed_jobs.append(job)

    def skip_remaining_jobs(self):
        while self.job_queue:
            job = self.job_queue.pop(0)
            self.skip_job(job)
        self.write_state()

    def write_state(self):
        self.run_output.write_state()

    def get_metric(self, name):
        try:
            return self.output.get_metric(name)
        except HostError:
            if not self.current_job:
                raise
            return self.run_output.get_metric(name)

    def add_metric(self, name, value, units=None, lower_is_better=False,
                   classifiers=None):
        if self.current_job:
            classifiers = merge_config_values(self.current_job.classifiers,
                                              classifiers)
        self.output.add_metric(name, value, units, lower_is_better, classifiers)

    def get_artifact(self, name):
        try:
            return self.output.get_artifact(name)
        except HostError:
            if not self.current_job:
                raise
            return self.run_output.get_artifact(name)

    def get_artifact_path(self, name):
        try:
            return self.output.get_artifact_path(name)
        except HostError:
            if not self.current_job:
                raise
            return self.run_output.get_artifact_path(name)

    def add_artifact(self, name, path, kind, description=None, classifiers=None):
        self.output.add_artifact(name, path, kind, description, classifiers)

    def add_run_artifact(self, name, path, kind, description=None,
                         classifiers=None):
        self.run_output.add_artifact(name, path, kind, description, classifiers)

    def add_event(self, message):
        self.output.add_event(message)

    def initialize_jobs(self):
        new_queue = []
        failed_ids = []
        for job in self.job_queue:
            if job.id in failed_ids:
                # Don't try to initialize a job if another job with the same ID
                # (i.e. same job spec) has failed - we can assume it will fail
                # too.
                self.skip_job(job)
                continue

            try:
                job.initialize(self)
            except WorkloadError as e:
                job.set_status(Status.FAILED)
                self.add_event(e.message)
                if not getattr(e, 'logged', None):
                    log.log_error(e, self.logger)
                    e.logged = True
                failed_ids.append(job.id)

                if self.cm.run_config.bail_on_init_failure:
                    raise
            else:
                new_queue.append(job)

        self.job_queue = new_queue


class Executor(object):
    """
    The ``Executor``'s job is to set up the execution context and pass to a
    ``Runner`` along with a loaded run specification. Once the ``Runner`` has
    done its thing, the ``Executor`` performs some final reporting before
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
        self.target_manager = None
        self.device = None

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
        self.target_manager = TargetManager(config.run_config.device,
                                       config.run_config.device_config,
                                       output.basepath)
        output.set_target_info(self.target_manager.get_target_info())

        self.logger.info('Initializing execution context')
        context = ExecutionContext(config_manager, self.target_manager, output)

        self.logger.info('Generating jobs')
        config_manager.generate_jobs(context)
        output.write_job_specs(config_manager.job_specs)
        output.write_state()

        self.logger.info('Installing instrumentation')
        for instrument in config_manager.get_instruments(self.target_manager.target):
            instrumentation.install(instrument, context)
        instrumentation.validate()

        self.logger.info('Installing result processors')
        pm = ProcessorManager()
        for proc in config_manager.get_processors():
            pm.install(proc, context)
        pm.validate()

        self.logger.info('Starting run')
        runner = Runner(context, pm)
        signal.send(signal.RUN_STARTED, self)
        runner.run()
        context.finalize()
        self.execute_postamble(context, output)
        signal.send(signal.RUN_COMPLETED, self)

    def execute_postamble(self, context, output):
        self.logger.info('Done.')
        duration = format_duration(output.info.duration)
        self.logger.info('Run duration: {}'.format(duration))
        num_ran = context.run_state.num_completed_jobs
        status_summary = 'Ran a total of {} iterations: '.format(num_ran)

        counter = context.run_state.get_status_counts()
        parts = []
        for status in reversed(Status.levels):
            if status in counter:
                parts.append('{} {}'.format(counter[status], status))
        self.logger.info(status_summary + ', '.join(parts))

        self.logger.info('Results can be found in {}'.format(output.basepath))

        if self.error_logged:
            self.logger.warn('There were errors during execution.')
            self.logger.warn('Please see {}'.format(output.logfile))
        elif self.warning_logged:
            self.logger.warn('There were warnings during execution.')
            self.logger.warn('Please see {}'.format(output.logfile))

    def _error_signalled_callback(self):
        self.error_logged = True
        signal.disconnect(self._error_signalled_callback, signal.ERROR_LOGGED)

    def _warning_signalled_callback(self):
        self.warning_logged = True
        signal.disconnect(self._warning_signalled_callback, signal.WARNING_LOGGED)


class Runner(object):
    """
    Triggers running jobs and processing results

    Takes pre-initialized ExcecutionContext and ProcessorManager. Handles
    actually running the jobs, and triggers the ProcessorManager to handle
    processing job and run results.
    """

    def __init__(self, context, pm):
        self.logger = logging.getLogger('runner')
        self.logger.context = context
        self.context = context
        self.pm = pm
        self.output = self.context.output
        self.config = self.context.cm

    def run(self):
        try:
            self.initialize_run()
            self.send(signal.RUN_INITIALIZED)

            while self.context.job_queue:
                try:
                    with signal.wrap('JOB_EXECUTION', self):
                        self.run_next_job(self.context)
                except KeyboardInterrupt:
                    self.context.skip_remaining_jobs()
        except Exception as e:
            self.context.add_event(e.message)
            if (not getattr(e, 'logged', None) and
                    not isinstance(e, KeyboardInterrupt)):
                log.log_error(e, self.logger)
                e.logged = True
            raise e
        finally:
            self.finalize_run()
            self.send(signal.RUN_FINALIZED)

    def initialize_run(self):
        self.logger.info('Initializing run')
        self.context.start_run()
        self.pm.initialize()
        log.indent()
        self.context.initialize_jobs()
        log.dedent()
        self.context.write_state()

    def finalize_run(self):
        self.logger.info('Finalizing run')
        self.context.end_run()
        self.pm.process_run_output(self.context)
        self.pm.export_run_output(self.context)
        self.pm.finalize()
        log.indent()
        for job in self.context.completed_jobs:
            job.finalize(self.context)
        log.dedent()

    def run_next_job(self, context):
        job = context.start_job()
        self.logger.info('Running job {}'.format(job.id))

        try:
            log.indent()
            self.do_run_job(job, context)
            job.set_status(Status.OK)
        except KeyboardInterrupt:
            job.set_status(Status.ABORTED)
            raise
        except Exception as e: # pylint: disable=broad-except
            job.set_status(Status.FAILED)
            context.add_event(e.message)
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
        rc = self.context.cm.run_config
        if job.workload.phones_home and not rc.allow_phone_home:
            self.logger.warning('Skipping job {} ({}) due to allow_phone_home=False'
                                .format(job.id, job.workload.name))
            self.context.skip_job(job)
            return

        job.set_status(Status.RUNNING)
        self.send(signal.JOB_STARTED)

        with signal.wrap('JOB_TARGET_CONFIG', self):
            job.configure_target(context)

        with signal.wrap('JOB_SETUP', self):
            job.setup(context)

        try:

            try:
                with signal.wrap('JOB_EXECUTION', self):
                    job.run(context)
            except Exception as e:
                job.set_status(Status.FAILED)
                if not getattr(e, 'logged', None):
                    log.log_error(e, self.logger)
                    e.logged = True
                raise e
            finally:
                try:
                    with signal.wrap('JOB_OUTPUT_PROCESSED', self):
                        job.process_output(context)
                    self.pm.process_job_output(context)
                    self.pm.export_job_output(context)
                except Exception:
                    job.set_status(Status.PARTIAL)
                    raise

        except KeyboardInterrupt:
            job.set_status(Status.ABORTED)
            self.logger.info('Got CTRL-C. Aborting.')
            raise
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
                self.retry_job(job)
                self.context.move_failed(job)
                self.context.write_state()
            else:
                msg = 'Job {} iteration {} completed with status {}. '\
                      'Max retries exceeded.'
                self.logger.error(msg.format(job.id, job.iteration, job.status))
                self.context.failed_jobs += 1
        else:  # status not in retry_on_status
            self.logger.info('Job completed with status {}'.format(job.status))
            self.context.successful_jobs += 1

    def retry_job(self, job):
        retry_job = Job(job.spec, job.iteration, self.context)
        retry_job.workload = job.workload
        retry_job.retries = job.retries + 1
        retry_job.set_status(Status.PENDING)
        self.context.job_queue.insert(0, retry_job)

    def send(self, s):
        signal.send(s, self, self.context)

    def __str__(self):
        return 'runner'
