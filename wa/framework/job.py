import logging
from datetime import datetime

from wa.framework import pluginloader, signal
from wa.framework.configuration.core import Status


class Job(object):

    _workload_cache = {}

    @property
    def id(self):
        return self.spec.id

    @property
    def label(self):
        return self.spec.label

    @property
    def classifiers(self):
        return self.spec.classifiers

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = value
        if self.output:
            self.output.status = value

    def __init__(self, spec, iteration, context):
        self.logger = logging.getLogger('job')
        self.spec = spec
        self.iteration = iteration
        self.context = context
        self.workload = None
        self.output = None
        self.run_time = None
        self.retries = 0
        self._status = Status.NEW

    def load(self, target, loader=pluginloader):
        self.logger.info('Loading job {}'.format(self.id))
        if self.iteration == 1:
            self.workload = loader.get_workload(self.spec.workload_name,
                                                target,
                                                **self.spec.workload_parameters)
            self.workload.init_resources(self.context)
            self.workload.validate()
            self._workload_cache[self.id] = self.workload
        else:
            self.workload = self._workload_cache[self.id]

    def initialize(self, context):
        self.logger.info('Initializing job {} [{}]'.format(self.id, self.iteration))
        with signal.wrap('WORKLOAD_INITIALIZED', self, context):
            self.workload.logger.context = context
            self.workload.initialize(context)
        self.set_status(Status.PENDING)
        context.update_job_state(self)

    def configure_target(self, context):
        self.logger.info('Configuring target for job {} [{}]'.format(self.id, self.iteration))
        context.tm.commit_runtime_parameters(self.spec.runtime_parameters)

    def setup(self, context):
        self.logger.info('Setting up job {} [{}]'.format(self.id, self.iteration))
        with signal.wrap('WORKLOAD_SETUP', self, context):
            self.workload.setup(context)

    def run(self, context):
        self.logger.info('Running job {} [{}]'.format(self.id, self.iteration))
        with signal.wrap('WORKLOAD_EXECUTION', self, context):
            start_time = datetime.utcnow()
            try:
                self.workload.run(context)
            finally:
                self.run_time = datetime.utcnow() - start_time

    def process_output(self, context):
        self.logger.info('Processing output for job {} [{}]'.format(self.id, self.iteration))
        if self.status != Status.FAILED:
            with signal.wrap('WORKLOAD_RESULT_EXTRACTION', self, context):
                self.workload.extract_results(context)
                context.extract_results()
            with signal.wrap('WORKLOAD_OUTPUT_UPDATE', self, context):
                self.workload.update_output(context)

    def teardown(self, context):
        self.logger.info('Tearing down job {} [{}]'.format(self.id, self.iteration))
        with signal.wrap('WORKLOAD_TEARDOWN', self, context):
            self.workload.teardown(context)

    def finalize(self, context):
        self.logger.info('Finalizing job {} [{}]'.format(self.id, self.iteration))
        with signal.wrap('WORKLOAD_FINALIZED', self, context):
            self.workload.finalize(context)

    def set_status(self, status, force=False):
        status = Status(status)
        if force or self.status < status:
            self.status = status
