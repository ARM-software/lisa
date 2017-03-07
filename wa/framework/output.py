import logging
import os
import shutil
import string
import sys
import uuid
from copy import copy

from wa.framework.configuration.core import JobSpec
from wa.framework.configuration.manager import ConfigManager
from wa.framework.target.info import TargetInfo
from wa.utils.misc import touch
from wa.utils.serializer import write_pod, read_pod


logger = logging.getLogger('output')


class RunInfo(object):
    """
    Information about the current run, such as its unique ID, run
    time, etc.

    """
    @staticmethod
    def from_pod(pod):
        uid = pod.pop('uuid')
        if uid is not None:
            uid = uuid.UUID(uid)
        instance = RunInfo(**pod)
        instance.uuid = uid
        return instance

    def __init__(self, run_name=None, project=None, project_stage=None,
                 start_time=None, end_time=None, duration=None):
        self.uuid = uuid.uuid4()
        self.run_name = None
        self.project = None
        self.project_stage = None
        self.start_time = None
        self.end_time = None
        self.duration = None

    def to_pod(self):
        d = copy(self.__dict__)
        d['uuid'] = str(self.uuid)
        return d


class RunState(object):
    """
    Represents the state of a WA run.

    """
    @staticmethod
    def from_pod(pod):
        return RunState()

    def __init__(self):
        pass

    def to_pod(self):
        return {}


class RunOutput(object):

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

    def __init__(self, path):
        self.basepath = path
        self.info = None
        self.state = None
        if (not os.path.isfile(self.statefile) or
                not os.path.isfile(self.infofile)):
            msg = '"{}" does not exist or is not a valid WA output directory.'
            raise ValueError(msg.format(self.basepath))
        self.reload()

    def reload(self):
        self.info = RunInfo.from_pod(read_pod(self.infofile))
        self.state = RunState.from_pod(read_pod(self.statefile))

    def write_info(self):
        write_pod(self.info.to_pod(), self.infofile)

    def write_state(self):
        write_pod(self.state.to_pod(), self.statefile)

    def write_config(self, config):
        write_pod(config.to_pod(), self.configfile)

    def read_config(self):
        if not os.path.isfile(self.configfile):
            return None
        return ConfigManager.from_pod(read_pod(self.configfile))

    def write_target_info(self, ti):
        write_pod(ti.to_pod(), self.targetfile)

    def read_config(self):
        if not os.path.isfile(self.targetfile):
            return None
        return TargetInfo.from_pod(read_pod(self.targetfile))

    def write_job_specs(self, job_specs):
        job_specs[0].to_pod()
        js_pod = {'jobs': [js.to_pod() for js in job_specs]}
        write_pod(js_pod, self.jobsfile)

    def read_job_specs(self):
        if not os.path.isfile(self.jobsfile):
            return None
        pod = read_pod(self.jobsfile)
        return [JobSpec.from_pod(jp) for jp in pod['jobs']]


def init_wa_output(path, wa_state, force=False):
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
    
    with open(os.path.join(path, '.run_state.json'), 'w') as wfh:
        wfh.write('{}')

    return RunOutput(path)


def _save_raw_config(meta_dir, state):
    raw_config_dir = os.path.join(meta_dir, 'raw_config')
    os.makedirs(raw_config_dir)

    for i, source in enumerate(state.loaded_config_sources):
        if not os.path.isfile(source):
            continue
        basename = os.path.basename(source)
        dest_path = os.path.join(raw_config_dir, 'cfg{}-{}'.format(i, basename))
        shutil.copy(source, dest_path)
                                     
                                     

