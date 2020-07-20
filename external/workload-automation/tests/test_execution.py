#    Copyright 2020 ARM Limited
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
import tempfile
from unittest import TestCase

from mock.mock import Mock
from nose.tools import assert_equal
from datetime import datetime

from wa.framework.configuration import RunConfiguration
from wa.framework.configuration.core import JobSpec, Status
from wa.framework.execution import ExecutionContext, Runner
from wa.framework.job import Job
from wa.framework.output import RunOutput, init_run_output
from wa.framework.output_processor import ProcessorManager
import wa.framework.signal as signal
from wa.framework.run import JobState


class MockConfigManager(Mock):

    @property
    def jobs(self):
        return self._joblist

    @property
    def loaded_config_sources(self):
        return []

    @property
    def run_config(self):
        return RunConfiguration()

    @property
    def plugin_cache(self):
        return MockPluginCache()

    def __init__(self, *args, **kwargs):
        super(MockConfigManager, self).__init__(*args, **kwargs)
        self._joblist = None
        self._run_config = RunConfiguration()

    def to_pod(self):
        return {}


class MockPluginCache(Mock):

    def list_plugins(self, kind=None):
        return []


class MockProcessorManager(Mock):

    def __init__(self, *args, **kwargs):
        super(MockProcessorManager, self).__init__(*args, **kwargs)

    def get_enabled(self):
        return []


class JobState_force_retry(JobState):

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if(self.retries != self.times_to_retry) and (value == Status.RUNNING):
            self._status = Status.FAILED
            if self.output:
                self.output.status = Status.FAILED
        else:
            self._status = value
            if self.output:
                self.output.status = value

    def __init__(self, to_retry, *args, **kwargs):
        self.retries = 0
        self._status = Status.NEW
        self.times_to_retry = to_retry
        self.output = None
        super(JobState_force_retry, self).__init__(*args, **kwargs)


class Job_force_retry(Job):
    '''This class imitates a job that retries as many times as specified by
    ``retries`` in its constructor'''

    def __init__(self, to_retry, *args, **kwargs):
        super(Job_force_retry, self).__init__(*args, **kwargs)
        self.state = JobState_force_retry(to_retry, self.id, self.label, self.iteration, Status.NEW)


class TestRunState(TestCase):

    def setUp(self):
        self.path = tempfile.mkstemp()[1]
        os.remove(self.path)
        self.initialise_signals()

        config = MockConfigManager()
        output = init_run_output(self.path, config)

        self.context = ExecutionContext(config, Mock(), output)

        self.job_spec = JobSpec()
        self.job_spec.augmentations = {}
        self.job_spec.finalize()

    def tearDown(self):
        signal.disconnect(self._verify_serialized_state, signal.RUN_INITIALIZED)
        signal.disconnect(self._verify_serialized_state, signal.JOB_STARTED)
        signal.disconnect(self._verify_serialized_state, signal.JOB_RESTARTED)
        signal.disconnect(self._verify_serialized_state, signal.JOB_COMPLETED)
        signal.disconnect(self._verify_serialized_state, signal.JOB_FAILED)
        signal.disconnect(self._verify_serialized_state, signal.JOB_ABORTED)
        signal.disconnect(self._verify_serialized_state, signal.RUN_FINALIZED)

    def test_job_state_transitions_pass(self):
        '''Tests state equality when the job passes first try'''
        job = Job(self.job_spec, 1, self.context)
        job.workload = Mock()

        self.context.cm._joblist = [job]
        self.context.run_state.add_job(job)

        runner = Runner(self.context, MockProcessorManager())
        runner.run()

    def test_job_state_transitions_fail(self):
        '''Tests state equality when job fails completely'''
        job = Job_force_retry(3, self.job_spec, 1, self.context)
        job.workload = Mock()

        self.context.cm._joblist = [job]
        self.context.run_state.add_job(job)

        runner = Runner(self.context, MockProcessorManager())
        runner.run()

    def test_job_state_transitions_retry(self):
        '''Tests state equality when job fails initially'''
        job = Job_force_retry(1, self.job_spec, 1, self.context)
        job.workload = Mock()

        self.context.cm._joblist = [job]
        self.context.run_state.add_job(job)

        runner = Runner(self.context, MockProcessorManager())
        runner.run()

    def initialise_signals(self):
        signal.connect(self._verify_serialized_state, signal.RUN_INITIALIZED)
        signal.connect(self._verify_serialized_state, signal.JOB_STARTED)
        signal.connect(self._verify_serialized_state, signal.JOB_RESTARTED)
        signal.connect(self._verify_serialized_state, signal.JOB_COMPLETED)
        signal.connect(self._verify_serialized_state, signal.JOB_FAILED)
        signal.connect(self._verify_serialized_state, signal.JOB_ABORTED)
        signal.connect(self._verify_serialized_state, signal.RUN_FINALIZED)

    def _verify_serialized_state(self, _):
        fs_state = RunOutput(self.path).state
        ex_state = self.context.run_output.state

        assert_equal(fs_state.status, ex_state.status)
        fs_js_zip = zip(
            [value for key, value in fs_state.jobs.items()],
            [value for key, value in ex_state.jobs.items()]
        )
        for fs_jobstate, ex_jobstate in fs_js_zip:
            assert_equal(fs_jobstate.iteration, ex_jobstate.iteration)
            assert_equal(fs_jobstate.retries, ex_jobstate.retries)
            assert_equal(fs_jobstate.status, ex_jobstate.status)


class TestJobState(TestCase):

    def setUp(self):
        path = tempfile.mkstemp()[1]
        os.remove(path)
        self.initialise_signals()

        config = MockConfigManager()
        output = init_run_output(path, config)

        self.context = ExecutionContext(config, Mock(), output)

    def test_job_retry_status(self):
        job_spec = JobSpec()
        job_spec.augmentations = {}
        job_spec.finalize()

        self.job = Job_force_retry(2, job_spec, 1, self.context)
        self.job.workload = Mock()

        self.context.cm._joblist = [self.job]
        self.context.run_state.add_job(self.job)

        runner = Runner(self.context, MockProcessorManager())
        runner.run()

    def initialise_signals(self):
        signal.connect(self._verify_restarted_job_status, signal.JOB_RESTARTED)

    def tearDown(self):
        signal.disconnect(self._verify_restarted_job_status, signal.JOB_RESTARTED)

    def _verify_restarted_job_status(self, _):
        assert_equal(self.job.status, Status.PENDING)
