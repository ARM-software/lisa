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
from collections import OrderedDict, Counter
from copy import copy
from datetime import datetime, timedelta

from wa.framework.configuration.core import Status

# Because of use of Enum (dynamic attrs)
# pylint: disable=no-member

class RunInfo(object):
    """
    Information about the current run, such as its unique ID, run
    time, etc.

    """
    @staticmethod
    def from_pod(pod):
        uid = pod.pop('uuid')
        duration = pod.pop('duration')
        if uid is not None:
            uid = uuid.UUID(uid)
        instance = RunInfo(**pod)
        instance.uuid = uid
        instance.duration = duration if duration is None else\
                            timedelta(seconds=duration)
        return instance

    def __init__(self, run_name=None, project=None, project_stage=None,
                 start_time=None, end_time=None, duration=None):
        self.uuid = uuid.uuid4()
        self.run_name = run_name
        self.project = project
        self.project_stage = project_stage
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration

    def to_pod(self):
        d = copy(self.__dict__)
        d['uuid'] = str(self.uuid)
        if self.duration is None:
            d['duration'] = self.duration
        else:
            d['duration'] = self.duration.total_seconds()
        return d


class RunState(object):
    """
    Represents the state of a WA run.

    """
    @staticmethod
    def from_pod(pod):
        instance = RunState()
        instance.status = Status(pod['status'])
        instance.timestamp = pod['timestamp']
        jss = [JobState.from_pod(j) for j in pod['jobs']]
        instance.jobs = OrderedDict(((js.id, js.iteration), js) for js in jss)
        return instance

    @property
    def num_completed_jobs(self):
        return sum(1 for js in self.jobs.itervalues()
                   if js.status > Status.RUNNING)

    def __init__(self):
        self.jobs = OrderedDict()
        self.status = Status.NEW
        self.timestamp = datetime.utcnow()

    def add_job(self, job):
        job_state = JobState(job.id, job.label, job.iteration, job.status)
        self.jobs[(job_state.id, job_state.iteration)] = job_state

    def update_job(self, job):
        state = self.jobs[(job.id, job.iteration)]
        state.status = job.status
        state.timestamp = datetime.utcnow()

    def get_status_counts(self):
        counter = Counter()
        for job_state in self.jobs.itervalues():
            counter[job_state.status] += 1
        return counter

    def to_pod(self):
        return OrderedDict(
            status=str(self.status),
            timestamp=self.timestamp,
            jobs=[j.to_pod() for j in self.jobs.itervalues()],
        )


class JobState(object):

    @staticmethod
    def from_pod(pod):
        instance = JobState(pod['id'], pod['label'], pod['iteration'], Status(pod['status']))
        instance.retries = pod['retries']
        instance.timestamp = pod['timestamp']
        return instance

    @property
    def output_name(self):
        return '{}-{}-{}'.format(self.id, self.label, self.iteration)

    def __init__(self, id, label, iteration, status):
        # pylint: disable=redefined-builtin
        self.id = id
        self.label = label
        self.iteration = iteration
        self.status = status
        self.retries = 0
        self.timestamp = datetime.utcnow()

    def to_pod(self):
        return OrderedDict(
            id=self.id,
            label=self.label,
            iteration=self.iteration,
            status=str(self.status),
            retries=0,
            timestamp=self.timestamp,
        )
