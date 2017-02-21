import os
import shutil
import logging
import uuid
from copy import copy
from datetime import datetime, timedelta

from wa.framework import signal, log
from wa.framework.configuration.core import merge_config_values
from wa.utils import serializer
from wa.utils.misc import enum_metaclass, ensure_directory_exists as _d
from wa.utils.types import numeric


class Status(object):

    __metaclass__ = enum_metaclass('values', return_name=True)

    values = [
        'NEW',
        'PENDING',
        'RUNNING',
        'COMPLETE',
        'OK',
        'OKISH',
        'NONCRITICAL',
        'PARTIAL',
        'FAILED',
        'ABORTED',
        'SKIPPED',
        'UNKNOWN',
    ]


class WAOutput(object):

    basename = '.wa-output'

    @classmethod
    def load(cls, source):
        if os.path.isfile(source):
            pod = serializer.load(source)
        elif os.path.isdir(source):
            pod = serializer.load(os.path.join(source, cls.basename))
        else:
            message = 'Cannot load {} from {}'
            raise ValueError(message.format(cls.__name__, source))
        return cls.from_pod(pod)

    @classmethod
    def from_pod(cls, pod):
        instance = cls(pod['output_directory'])
        instance.status = pod['status']
        instance.metrics = [Metric.from_pod(m) for m in pod['metrics']]
        instance.artifacts = [Artifact.from_pod(a) for a in pod['artifacts']]
        instance.events = [RunEvent.from_pod(e) for e in pod['events']]
        instance.classifiers = pod['classifiers']
        return instance

    def __init__(self, output_directory):
        self.logger = logging.getLogger('output')
        self.output_directory = output_directory
        self.status = Status.UNKNOWN
        self.classifiers = {}
        self.metrics = []
        self.artifacts = []
        self.events = []
        
    def initialize(self, overwrite=False):
        if os.path.exists(self.output_directory):
            if not overwrite:
                raise RuntimeError('"{}" already exists.'.format(self.output_directory))
            self.logger.info('Removing existing output directory.')
            shutil.rmtree(self.output_directory)
        self.logger.debug('Creating output directory {}'.format(self.output_directory))
        os.makedirs(self.output_directory)

    def add_metric(self, name, value, units=None, lower_is_better=False, classifiers=None):
        classifiers = merge_config_values(self.classifiers, classifiers or {})
        self.metrics.append(Metric(name, value, units, lower_is_better, classifiers))

    def add_artifact(self, name, path, kind, *args, **kwargs):
        path = _check_artifact_path(path, self.output_directory)
        self.artifacts.append(Artifact(name, path, kind, Artifact.RUN, *args, **kwargs))

    def get_path(self, subpath):
        return os.path.join(self.output_directory, subpath)

    def to_pod(self):
        return {
            'output_directory': self.output_directory,
            'status': self.status,
            'metrics': [m.to_pod() for m in self.metrics],
            'artifacts': [a.to_pod() for a in self.artifacts],
            'events': [e.to_pod() for e in self.events],
            'classifiers': copy(self.classifiers),
        }

    def persist(self):
        statefile = os.path.join(self.output_directory, self.basename)
        with open(statefile, 'wb') as wfh:
            serializer.dump(self, wfh)
        

class RunInfo(object):

    default_name_format = 'wa-run-%y%m%d-%H%M%S'

    def __init__(self, project=None, project_stage=None, name=None):
        self.uuid = uuid.uuid4()
        self.project = project
        self.project_stage = project_stage
        self.name = name or datetime.now().strftime(self.default_name_format)
        self.start_time = None
        self.end_time = None
        self.duration = None

    @staticmethod
    def from_pod(pod):
        instance = RunInfo()
        instance.uuid = uuid.UUID(pod['uuid'])
        instance.project = pod['project']
        instance.project_stage = pod['project_stage']
        instance.name = pod['name']
        instance.start_time = pod['start_time']
        instance.end_time = pod['end_time']
        instance.duration = timedelta(seconds=pod['duration'])
        return instance

    def to_pod(self):
        d = copy(self.__dict__)
        d['uuid'] = str(self.uuid)
        d['duration'] = self.duration.days * 3600 * 24 + self.duration.seconds
        return d


class RunOutput(WAOutput):

    @property
    def info_directory(self):
        return _d(os.path.join(self.output_directory, '_info'))

    @property
    def config_directory(self):
        return _d(os.path.join(self.output_directory, '_config'))

    @property
    def failed_directory(self):
        return _d(os.path.join(self.output_directory, '_failed'))

    @property
    def log_file(self):
        return os.path.join(self.output_directory, 'run.log')

    @classmethod
    def from_pod(cls, pod):
        instance = WAOutput.from_pod(pod)
        instance.info = RunInfo.from_pod(pod['info'])
        instance.jobs = [JobOutput.from_pod(i) for i in pod['jobs']]
        instance.failed = [JobOutput.from_pod(i) for i in pod['failed']]
        return instance

    def __init__(self, output_directory):
        super(RunOutput, self).__init__(output_directory)
        self.logger = logging.getLogger('output')
        self.info = RunInfo()
        self.jobs = []
        self.failed = []

    def initialize(self, overwrite=False):
        super(RunOutput, self).initialize(overwrite)
        log.add_file(self.log_file)
        self.add_artifact('runlog', self.log_file,  'log')

    def create_job_output(self, id):
        outdir = os.path.join(self.output_directory, id)
        job_output = JobOutput(outdir)
        self.jobs.append(job_output)
        return job_output

    def move_failed(self, job_output):
        basename = os.path.basename(job_output.output_directory)
        i = 1
        dest = os.path.join(self.failed_directory, basename + '-{}'.format(i))
        while os.path.exists(dest):
            i += 1
            dest = '{}-{}'.format(dest[:-2], i)
        shutil.move(job_output.output_directory, dest)

    def to_pod(self):
        pod = super(RunOutput, self).to_pod()
        pod['info'] = self.info.to_pod()
        pod['jobs'] = [i.to_pod() for i in self.jobs]
        pod['failed'] = [i.to_pod() for i in self.failed]
        return pod


class JobOutput(WAOutput):

    def add_artifact(self, name, path, kind, *args, **kwargs):
        path = _check_artifact_path(path, self.output_directory)
        self.artifacts.append(Artifact(name, path, kind, Artifact.ITERATION, *args, **kwargs))


class Artifact(object):
    """
    This is an artifact generated during execution/post-processing of a workload.
    Unlike metrics, this represents an actual artifact, such as a file, generated.
    This may be "result", such as trace, or it could be "meta data" such as logs.
    These are distinguished using the ``kind`` attribute, which also helps WA decide
    how it should be handled. Currently supported kinds are:

        :log: A log file. Not part of "results" as such but contains information about the
              run/workload execution that be useful for diagnostics/meta analysis.
        :meta: A file containing metadata. This is not part of "results", but contains
               information that may be necessary to reproduce the results (contrast with
               ``log`` artifacts which are *not* necessary).
        :data: This file contains new data, not available otherwise and should be considered
               part of the "results" generated by WA. Most traces would fall into this category.
        :export: Exported version of results or some other artifact. This signifies that
                 this artifact does not contain any new data that is not available
                 elsewhere and that it may be safely discarded without losing information.
        :raw: Signifies that this is a raw dump/log that is normally processed to extract
              useful information and is then discarded. In a sense, it is the opposite of
              ``export``, but in general may also be discarded.

              .. note:: whether a file is marked as ``log``/``data`` or ``raw`` depends on
                        how important it is to preserve this file, e.g. when archiving, vs
                        how much space it takes up. Unlike ``export`` artifacts which are
                        (almost) always ignored by other exporters as that would never result
                        in data loss, ``raw`` files *may* be processed by exporters if they
                        decided that the risk of losing potentially (though unlikely) useful
                        data is greater than the time/space cost of handling the artifact (e.g.
                        a database uploader may choose to ignore ``raw`` artifacts, where as a
                        network filer archiver may choose to archive them).

        .. note: The kind parameter is intended to represent the logical function of a particular
                 artifact, not it's intended means of processing -- this is left entirely up to the
                 result processors.

    """

    RUN = 'run'
    ITERATION = 'iteration'

    valid_kinds = ['log', 'meta', 'data', 'export', 'raw']

    @staticmethod
    def from_pod(pod):
        return Artifact(**pod)

    def __init__(self, name, path, kind, level=RUN, mandatory=False, description=None):
        """"
        :param name: Name that uniquely identifies this artifact.
        :param path: The *relative* path of the artifact. Depending on the ``level``
                     must be either relative to the run or iteration output directory.
                     Note: this path *must* be delimited using ``/`` irrespective of the
                     operating system.
        :param kind: The type of the artifact this is (e.g. log file, result, etc.) this
                     will be used a hit to result processors. This must be one of ``'log'``,
                     ``'meta'``, ``'data'``, ``'export'``, ``'raw'``.
        :param level: The level at which the artifact will be generated. Must be either
                      ``'iteration'`` or ``'run'``.
        :param mandatory: Boolean value indicating whether this artifact must be present
                          at the end of result processing for its level.
        :param description: A free-form description of what this artifact is.

        """
        if kind not in self.valid_kinds:
            raise ValueError('Invalid Artifact kind: {}; must be in {}'.format(kind, self.valid_kinds))
        self.name = name
        self.path = path.replace('/', os.sep) if path is not None else path
        self.kind = kind
        self.level = level
        self.mandatory = mandatory
        self.description = description

    def exists(self, context):
        """Returns ``True`` if artifact exists within the specified context, and
        ``False`` otherwise."""
        fullpath = os.path.join(context.output_directory, self.path)
        return os.path.exists(fullpath)

    def to_pod(self):
        return copy(self.__dict__)


class RunEvent(object):
    """
    An event that occured during a run.

    """

    @staticmethod
    def from_pod(pod):
        instance = RunEvent(pod['message'])
        instance.timestamp = pod['timestamp']
        return instance

    def __init__(self, message):
        self.timestamp = datetime.utcnow()
        self.message = message

    def to_pod(self):
        return copy(self.__dict__)

    def __str__(self):
        return '{} {}'.format(self.timestamp, self.message)

    __repr__ = __str__


class Metric(object):
    """
    This is a single metric collected from executing a workload.

    :param name: the name of the metric. Uniquely identifies the metric
                 within the results.
    :param value: The numerical value of the metric for this execution of
                  a workload. This can be either an int or a float.
    :param units: Units for the collected value. Can be None if the value
                  has no units (e.g. it's a count or a standardised score).
    :param lower_is_better: Boolean flag indicating where lower values are
                            better than higher ones. Defaults to False.
    :param classifiers: A set of key-value pairs to further classify this metric
                        beyond current iteration (e.g. this can be used to identify
                        sub-tests).

    """

    @staticmethod
    def from_pod(pod):
        return Metric(**pod)

    def __init__(self, name, value, units=None, lower_is_better=False, classifiers=None):
        self.name = name
        self.value = numeric(value)
        self.units = units
        self.lower_is_better = lower_is_better
        self.classifiers = classifiers or {}

    def to_pod(self):
        return copy(self.__dict__)

    def __str__(self):
        result = '{}: {}'.format(self.name, self.value)
        if self.units:
            result += ' ' + self.units
        result += ' ({})'.format('-' if self.lower_is_better else '+')
        return '<{}>'.format(result)

    __repr__ = __str__


def _check_artifact_path(path, rootpath):
    if path.startswith(rootpath):
        return os.path.abspath(path)
    rootpath = os.path.abspath(rootpath)
    full_path = os.path.join(rootpath, path)
    if not os.path.isfile(full_path):
        raise ValueError('Cannot add artifact because {} does not exist.'.format(full_path))
    return full_path
