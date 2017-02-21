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
import uuid
import logging
from copy import copy
from datetime import datetime, timedelta
from collections import OrderedDict

from wa.framework import signal, pluginloader, log
from wa.framework.plugin import Plugin
from wa.framework.output import Status
from wa.framework.resource import ResourceResolver
from wa.framework.exception import JobError
from wa.utils import counter
from wa.utils.serializer import json
from wa.utils.misc import ensure_directory_exists as _d
from wa.utils.types import TreeNode, caseless_string



class JobActor(object):

    def get_config(self):
        return {}

    def initialize(self, context):
        pass

    def run(self):
        pass

    def finalize(self):
        pass

    def restart(self):
        pass

    def complete(self):
        pass


class RunnerJob(object):

    @property
    def status(self):
        return self.output.status

    @status.setter
    def status(self, value):
        self.output.status = value

    @property
    def should_retry(self):
        return self.attempt <= self.max_retries

    def __init__(self, id, actor, output, max_retries):
        self.id = id
        self.actor = actor
        self.output = output
        self.max_retries = max_retries
        self.status = Status.NEW
        self.attempt = 0

    def initialize(self, context):
        self.actor.initialize(context)
        self.status = Status.PENDING

    def run(self):
        self.status = Status.RUNNING
        self.attempt += 1
        self.output.config = self.actor.get_config()
        self.output.initialize()
        self.actor.run()
        self.status = Status.COMPLETE

    def finalize(self):
        self.actor.finalize()

    def restart(self):
        self.actor.restart()

    def complete(self):
        self.actor.complete()


__run_methods = set()


def runmethod(method):
    """
    A method decorator that ensures that a method is invoked only once per run.

    """
    def _method_wrapper(*args, **kwargs):
        if method in __run_methods:
            return
        __run_methods.add(method)
        ret = method(*args, **kwargs)
        if ret is not None:
            message = 'runmethod()\'s must return None; method "{}" returned "{}"'
            raise RuntimeError(message.format(method, ret))
    return _method_wrapper


def reset_runmethods():
    global __run_methods
    __run_methods = set()


class Runner(object):

    @property
    def info(self):
        return self.output.info

    @property
    def status(self):
        return self.output.status

    @status.setter
    def status(self, value):
        self.output.status = value

    @property
    def jobs_pending(self):
        return len(self.job_queue) > 0

    @property
    def current_job(self):
        if self.job_queue:
            return self.job_queue[0]

    @property
    def previous_job(self):
        if self.completed_jobs:
            return self.completed_jobs[-1]

    @property
    def next_job(self):
        if len(self.job_queue) > 1:
            return self.job_queue[1]

    def __init__(self, output):
        self.logger = logging.getLogger('runner')
        self.output = output
        self.context = RunContext(self)
        self.status = Status.NEW
        self.job_queue = []
        self.completed_jobs = []
        self._known_ids = set([])

    def add_job(self, job_id, actor, max_retries=2):
        job_id = caseless_string(job_id)
        if job_id in self._known_ids:
            raise JobError('Job with id "{}" already exists'.format(job_id))
        output = self.output.create_job_output(job_id)
        self.job_queue.append(RunnerJob(job_id, actor, output, max_retries))
        self._known_ids.add(job_id)

    def initialize(self):
        self.logger.info('Initializing run')
        self.start_time = datetime.now()
        if not self.info.start_time:
            self.info.start_time = self.start_time
            self.info.duration = timedelta()

        self.context.initialize()
        for job in self.job_queue:
            job.initialize(self.context)
        self.persist_state()
        self.logger.info('Run initialized')

    def run(self):
        self.status = Status.RUNNING
        reset_runmethods()
        signal.send(signal.RUN_STARTED, self, self.context)
        self.initialize()
        signal.send(signal.RUN_INITIALIZED, self, self.context)
        self.run_jobs()
        signal.send(signal.RUN_COMPLETED, self, self.context)
        self.finalize()
        signal.send(signal.RUN_FINALIZED, self, self.context)

    def run_jobs(self):
        try:
            self.logger.info('Running jobs')
            while self.jobs_pending:
                self.begin_job()
                log.indent()
                try:
                    self.current_job.run()
                except KeyboardInterrupt:
                    self.current_job.status = Status.ABORTED
                    signal.send(signal.JOB_ABORTED, self, self.current_job)
                    raise
                except Exception as e:
                    self.current_job.status = Status.FAILED
                    log.log_error(e, self.logger)
                    signal.send(signal.JOB_FAILED, self, self.current_job)
                else:
                    self.current_job.status = Status.COMPLETE
                finally:
                    log.dedent()
                    self.complete_job()
        except KeyboardInterrupt:
            self.status = Status.ABORTED
            while self.job_queue:
                job = self.job_queue.pop(0)
                job.status = RunnerJob.ABORTED
                self.completed_jobs.append(job)
            signal.send(signal.RUN_ABORTED, self, self)
            raise
        except Exception as e:
            self.status = Status.FAILED
            log.log_error(e, self.logger)
            signal.send(signal.RUN_FAILED, self, self)
        else:
            self.status = Status.COMPLETE

    def finalize(self):
        self.logger.info('Finalizing run')
        for job in self.job_queue:
            job.finalize()
        self.end_time = datetime.now()
        self.info.end_time = self.end_time
        self.info.duration += self.end_time - self.start_time
        self.persist_state()
        signal.send(signal.RUN_FINALIZED, self, self)
        self.logger.info('Run completed')

    def begin_job(self):
        self.logger.info('Starting job {}'.format(self.current_job.id))
        signal.send(signal.JOB_STARTED, self, self.current_job)
        self.persist_state()

    def complete_job(self):
        if self.current_job.status == Status.FAILED:
            self.output.move_failed(self.current_job.output)
            if self.current_job.should_retry:
                self.logger.info('Restarting job {}'.format(self.current_job.id))
                self.persist_state()
                self.current_job.restart()
                signal.send(signal.JOB_RESTARTED, self, self.current_job)
                return

        self.logger.info('Completing job {}'.format(self.current_job.id))
        self.current_job.complete()
        self.persist_state()
        signal.send(signal.JOB_COMPLETED, self, self.current_job)
        job = self.job_queue.pop(0)
        self.completed_jobs.append(job)

    def persist_state(self):
        self.output.persist()


class RunContext(object):
    """
    Provides a context for instrumentation. Keeps track of things like
    current workload and iteration.

    """

    @property
    def run_output(self):
        return self.runner.output

    @property
    def current_job(self):
        return self.runner.current_job

    @property
    def run_output_directory(self):
        return self.run_output.output_directory

    @property
    def output_directory(self):
        if self.runner.current_job:
            return self.runner.current_job.output.output_directory
        else:
            return self.run_output.output_directory

    @property
    def info_directory(self):
        return self.run_output.info_directory

    @property
    def config_directory(self):
        return self.run_output.config_directory

    @property
    def failed_directory(self):
        return self.run_output.failed_directory

    @property
    def log_file(self):
        return os.path.join(self.output_directory, 'run.log')


    def __init__(self, runner):
        self.runner = runner
        self.job = None
        self.iteration = None
        self.job_output = None
        self.resolver = ResourceResolver()

    def initialize(self):
        self.resolver.load()

    def get_path(self, subpath):
        if self.current_job is None:
            return self.run_output.get_path(subpath)
        else:
            return self.current_job.output.get_path(subpath)

    def add_metric(self, *args, **kwargs):
        if self.current_job is None:
            self.run_output.add_metric(*args, **kwargs)
        else:
            self.current_job.output.add_metric(*args, **kwargs)

    def add_artifact(self, name, path, kind, *args, **kwargs):
        if self.current_job is None:
            self.add_run_artifact(name, path, kind, *args, **kwargs)
        else:
            self.add_job_artifact(name, path, kind, *args, **kwargs)

    def add_run_artifact(self, *args, **kwargs):
        self.run_output.add_artifiact(*args, **kwargs)

    def add_job_artifact(self, *args, **kwargs):
        self.current_job.output.add_artifact(*args, **kwargs)

    def get_artifact(self, name):
        if self.iteration_artifacts:
            for art in self.iteration_artifacts:
                if art.name == name:
                    return art
        for art in self.run_artifacts:
            if art.name == name:
                return art
        return None

