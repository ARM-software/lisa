import logging
import os
import shutil
from copy import copy
from datetime import datetime

from wa.framework.configuration.core import JobSpec, Status
from wa.framework.configuration.execution import CombinedConfig
from wa.framework.exception import HostError
from wa.framework.run import RunState, RunInfo
from wa.framework.target.info import TargetInfo
from wa.utils.misc import touch, ensure_directory_exists
from wa.utils.serializer import write_pod, read_pod
from wa.utils.types import enum, numeric


logger = logging.getLogger('output')


class Output(object):

    kind = None

    @property
    def resultfile(self):
        return os.path.join(self.basepath, 'result.json')

    @property
    def event_summary(self):
        num_events = len(self.events)
        if num_events:
            lines = self.events[0].message.split('\n')
            message = '({} event(s)): {}'
            if num_events > 1 or len(lines) > 1:
                message += '[...]'
            return message.format(num_events, lines[0])
        return ''

    @property
    def status(self):
        if self.result is None:
            return None
        return self.result.status

    @status.setter
    def status(self, value):
        self.result.status = value

    @property
    def metrics(self):
        if self.result is None:
            return []
        return self.result.metrics

    @property
    def artifacts(self):
        if self.result is None:
            return []
        return self.result.artifacts

    def __init__(self, path):
        self.basepath = path
        self.result = None
        self.events = []

    def reload(self):
        pod = read_pod(self.resultfile)
        self.result = Result.from_pod(pod)

    def write_result(self):
        write_pod(self.result.to_pod(), self.resultfile)

    def get_path(self, subpath):
        return os.path.join(self.basepath, subpath.strip(os.sep))

    def add_metric(self, name, value, units=None, lower_is_better=False,
                   classifiers=None):
        self.result.add_metric(name, value, units, lower_is_better, classifiers)

    def add_artifact(self, name, path, kind, description=None, classifiers=None):
        if not os.path.exists(path):
            path = self.get_path(path)
        if not os.path.exists(path):
            msg = 'Attempting to add non-existing artifact: {}'
            raise HostError(msg.format(path))
        path = os.path.relpath(path, self.basepath)

        self.result.add_artifact(name, path, kind, description, classifiers)

    def add_event(self, message):
        self.result.add_event(message)

    def get_metric(self, name):
        return self.result.get_metric(name)

    def get_artifact(self, name):
        return self.result.get_artifact(name)

    def get_artifact_path(self, name):
        artifact = self.get_artifact(name)
        return self.get_path(artifact.path)


class RunOutput(Output):

    kind = 'run'

    @property
    def logfile(self):
        return os.path.join(self.basepath, 'run.log')

    @property
    def metadir(self):
        return os.path.join(self.basepath, '__meta')

    @property
    def infofile(self):
        return os.path.join(self.metadir, 'run_info.json')

    @property
    def statefile(self):
        return os.path.join(self.basepath, '.run_state.json')

    @property
    def configfile(self):
        return os.path.join(self.metadir, 'config.json')

    @property
    def targetfile(self):
        return os.path.join(self.metadir, 'target_info.json')

    @property
    def jobsfile(self):
        return os.path.join(self.metadir, 'jobs.json')

    @property
    def raw_config_dir(self):
        return os.path.join(self.metadir, 'raw_config')

    @property
    def failed_dir(self):
        path = os.path.join(self.basepath, '__failed')
        return ensure_directory_exists(path)

    def __init__(self, path):
        super(RunOutput, self).__init__(path)
        self.info = None
        self.state = None
        self.result = None
        self.target_info = None
        self.jobs = []
        if (not os.path.isfile(self.statefile) or
                not os.path.isfile(self.infofile)):
            msg = '"{}" does not exist or is not a valid WA output directory.'
            raise ValueError(msg.format(self.basepath))
        self.reload()

    def reload(self):
        super(RunOutput, self).reload()
        self.info = RunInfo.from_pod(read_pod(self.infofile))
        self.state = RunState.from_pod(read_pod(self.statefile))
        if os.path.isfile(self.targetfile):
            self.target_info = TargetInfo.from_pod(read_pod(self.targetfile))

        for job_state in self.state.jobs.itervalues():
            job_path = os.path.join(self.basepath, job_state.output_name)
            job = JobOutput(job_path, job_state.id,
                            job_state.label, job_state.iteration,
                            job_state.retries)
            self.jobs.append(job)

    def write_info(self):
        write_pod(self.info.to_pod(), self.infofile)

    def write_state(self):
        write_pod(self.state.to_pod(), self.statefile)

    def write_config(self, config):
        write_pod(config.to_pod(), self.configfile)

    def read_config(self):
        if not os.path.isfile(self.configfile):
            return None
        return CombinedConfig.from_pod(read_pod(self.configfile))

    def set_target_info(self, ti):
        self.target_info = ti
        write_pod(ti.to_pod(), self.targetfile)

    def write_job_specs(self, job_specs):
        job_specs[0].to_pod()
        js_pod = {'jobs': [js.to_pod() for js in job_specs]}
        write_pod(js_pod, self.jobsfile)

    def read_job_specs(self):
        if not os.path.isfile(self.jobsfile):
            return None
        pod = read_pod(self.jobsfile)
        return [JobSpec.from_pod(jp) for jp in pod['jobs']]

    def move_failed(self, job_output):
        name = os.path.basename(job_output.basepath)
        attempt = job_output.retry + 1
        failed_name = '{}-attempt{:02}'.format(name, attempt)
        failed_path = os.path.join(self.failed_dir, failed_name)
        if os.path.exists(failed_path):
            raise ValueError('Path {} already exists'.format(failed_path))
        shutil.move(job_output.basepath, failed_path)
        job_output.basepath = failed_path


class JobOutput(Output):

    kind = 'job'

    # pylint: disable=redefined-builtin
    def __init__(self, path, id, label, iteration, retry):
        super(JobOutput, self).__init__(path)
        self.id = id
        self.label = label
        self.iteration = iteration
        self.retry = retry
        self.result = None
        self.reload()


class Result(object):

    @staticmethod
    def from_pod(pod):
        instance = Result()
        instance.status = Status(pod['status'])
        instance.metrics = [Metric.from_pod(m) for m in pod['metrics']]
        instance.artifacts = [Artifact.from_pod(a) for a in pod['artifacts']]
        instance.events = [Event.from_pod(e) for e in pod['events']]
        return instance

    def __init__(self):
        # pylint: disable=no-member
        self.status = Status.NEW
        self.metrics = []
        self.artifacts = []
        self.events = []

    def add_metric(self, name, value, units=None, lower_is_better=False,
                   classifiers=None):
        metric = Metric(name, value, units, lower_is_better, classifiers)
        logger.debug('Adding metric: {}'.format(metric))
        self.metrics.append(metric)

    def add_artifact(self, name, path, kind, description=None, classifiers=None):
        artifact = Artifact(name, path, kind, description=description,
                            classifiers=classifiers)
        logger.debug('Adding artifact: {}'.format(artifact))
        self.artifacts.append(artifact)

    def add_event(self, message):
        self.events.append(Event(message))

    def get_metric(self, name):
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def get_artifact(self, name):
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        raise HostError('Artifact "{}" not found'.format(name))

    def to_pod(self):
        return dict(
            status=str(self.status),
            metrics=[m.to_pod() for m in self.metrics],
            artifacts=[a.to_pod() for a in self.artifacts],
            events=[e.to_pod() for e in self.events],
        )


ARTIFACT_TYPES = ['log', 'meta', 'data', 'export', 'raw']
ArtifactType = enum(ARTIFACT_TYPES)


class Artifact(object):
    """
    This is an artifact generated during execution/post-processing of a
    workload.  Unlike metrics, this represents an actual artifact, such as a
    file, generated.  This may be "result", such as trace, or it could be "meta
    data" such as logs.  These are distinguished using the ``kind`` attribute,
    which also helps WA decide how it should be handled. Currently supported
    kinds are:

        :log: A log file. Not part of "results" as such but contains
              information about the run/workload execution that be useful for
              diagnostics/meta analysis.
        :meta: A file containing metadata. This is not part of "results", but
               contains information that may be necessary to reproduce the
               results (contrast with ``log`` artifacts which are *not*
               necessary).
        :data: This file contains new data, not available otherwise and should
               be considered part of the "results" generated by WA. Most traces
               would fall into this category.
        :export: Exported version of results or some other artifact. This
                 signifies that this artifact does not contain any new data
                 that is not available elsewhere and that it may be safely
                 discarded without losing information.
        :raw: Signifies that this is a raw dump/log that is normally processed
              to extract useful information and is then discarded. In a sense,
              it is the opposite of ``export``, but in general may also be
              discarded.

              .. note:: whether a file is marked as ``log``/``data`` or ``raw``
                        depends on how important it is to preserve this file,
                        e.g. when archiving, vs how much space it takes up.
                        Unlike ``export`` artifacts which are (almost) always
                        ignored by other exporters as that would never result
                        in data loss, ``raw`` files *may* be processed by
                        exporters if they decided that the risk of losing
                        potentially (though unlikely) useful data is greater
                        than the time/space cost of handling the artifact (e.g.
                        a database uploader may choose to ignore ``raw``
                        artifacts, where as a network filer archiver may choose
                        to archive them).

        .. note: The kind parameter is intended to represent the logical
                 function of a particular artifact, not it's intended means of
                 processing -- this is left entirely up to the result
                 processors.

    """

    @staticmethod
    def from_pod(pod):
        pod['kind'] = ArtifactType(pod['kind'])
        return Artifact(**pod)

    def __init__(self, name, path, kind, description=None, classifiers=None):
        """"
        :param name: Name that uniquely identifies this artifact.
        :param path: The *relative* path of the artifact. Depending on the
                     ``level`` must be either relative to the run or iteration
                     output directory.  Note: this path *must* be delimited
                     using ``/`` irrespective of the
                     operating system.
        :param kind: The type of the artifact this is (e.g. log file, result,
                     etc.) this will be used a hit to result processors. This
                     must be one of ``'log'``, ``'meta'``, ``'data'``,
                     ``'export'``, ``'raw'``.
        :param description: A free-form description of what this artifact is.
        :param classifiers: A set of key-value pairs to further classify this
                            metric beyond current iteration (e.g. this can be
                            used to identify sub-tests).

        """
        self.name = name
        self.path = path.replace('/', os.sep) if path is not None else path
        try:
            self.kind = ArtifactType(kind)
        except ValueError:
            msg = 'Invalid Artifact kind: {}; must be in {}'
            raise ValueError(msg.format(kind, ARTIFACT_TYPES))
        self.description = description
        self.classifiers = classifiers or {}

    def to_pod(self):
        pod = copy(self.__dict__)
        pod['kind'] = str(self.kind)
        return pod

    def __str__(self):
        return self.path

    def __repr__(self):
        return '{} ({}): {}'.format(self.name, self.kind, self.path)


class Metric(object):
    """
    This is a single metric collected from executing a workload.

    :param name: the name of the metric. Uniquely identifies the metric
                 within the results.
    :param value: The numerical value of the metric for this execution of a
                  workload. This can be either an int or a float.
    :param units: Units for the collected value. Can be None if the value
                  has no units (e.g. it's a count or a standardised score).
    :param lower_is_better: Boolean flag indicating where lower values are
                            better than higher ones. Defaults to False.
    :param classifiers: A set of key-value pairs to further classify this
                        metric beyond current iteration (e.g. this can be used
                        to identify sub-tests).

    """

    __slots__ = ['name', 'value', 'units', 'lower_is_better', 'classifiers']

    @staticmethod
    def from_pod(pod):
        return Metric(**pod)

    def __init__(self, name, value, units=None, lower_is_better=False,
                 classifiers=None):
        self.name = name
        self.value = numeric(value)
        self.units = units
        self.lower_is_better = lower_is_better
        self.classifiers = classifiers or {}

    def to_pod(self):
        return dict(
            name=self.name,
            value=self.value,
            units=self.units,
            lower_is_better=self.lower_is_better,
            classifiers=self.classifiers,
        )

    def __str__(self):
        result = '{}: {}'.format(self.name, self.value)
        if self.units:
            result += ' ' + self.units
        result += ' ({})'.format('-' if self.lower_is_better else '+')
        return result

    def __repr__(self):
        text = self.__str__()
        if self.classifiers:
            return '<{} {}>'.format(text, self.classifiers)
        else:
            return '<{}>'.format(text)


class Event(object):
    """
    An event that occured during a run.

    """

    __slots__ = ['timestamp', 'message']

    @staticmethod
    def from_pod(pod):
        instance = Event(pod['message'])
        instance.timestamp = pod['timestamp']
        return instance

    @property
    def summary(self):
        lines = self.message.split('\n')
        result = lines[0]
        if len(lines) > 1:
            result += '[...]'
        return result

    def __init__(self, message):
        self.timestamp = datetime.utcnow()
        self.message = message

    def to_pod(self):
        return dict(
            timestamp=self.timestamp,
            message=self.message,
        )

    def __str__(self):
        return '[{}] {}'.format(self.timestamp, self.message)

    __repr__ = __str__


def init_run_output(path, wa_state, force=False):
    if os.path.exists(path):
        if force:
            logger.info('Removing existing output directory.')
            shutil.rmtree(os.path.abspath(path))
        else:
            raise RuntimeError('path exists: {}'.format(path))

    logger.info('Creating output directory.')
    os.makedirs(path)
    meta_dir = os.path.join(path, '__meta')
    os.makedirs(meta_dir)
    _save_raw_config(meta_dir, wa_state)
    touch(os.path.join(path, 'run.log'))

    info = RunInfo(
            run_name=wa_state.run_config.run_name,
            project=wa_state.run_config.project,
            project_stage=wa_state.run_config.project_stage,
           )
    write_pod(info.to_pod(), os.path.join(meta_dir, 'run_info.json'))
    write_pod(RunState().to_pod(), os.path.join(path, '.run_state.json'))
    write_pod(Result().to_pod(), os.path.join(path, 'result.json'))

    return RunOutput(path)


def init_job_output(run_output, job):
    output_name = '{}-{}-{}'.format(job.id, job.spec.label, job.iteration)
    path = os.path.join(run_output.basepath, output_name)
    ensure_directory_exists(path)
    write_pod(Result().to_pod(), os.path.join(path, 'result.json'))
    job_output = JobOutput(path, job.id, job.label, job.iteration, job.retries)
    job_output.status = job.status
    run_output.jobs.append(job_output)
    return job_output


def discover_wa_outputs(path):
    for root, dirs, files in os.walk(path):
        if '__meta' in dirs:
            yield  RunOutput(root)


def _save_raw_config(meta_dir, state):
    raw_config_dir = os.path.join(meta_dir, 'raw_config')
    os.makedirs(raw_config_dir)

    for i, source in enumerate(state.loaded_config_sources):
        if not os.path.isfile(source):
            continue
        basename = os.path.basename(source)
        dest_path = os.path.join(raw_config_dir, 'cfg{}-{}'.format(i, basename))
        shutil.copy(source, dest_path)
