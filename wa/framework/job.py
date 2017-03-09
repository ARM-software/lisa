import logging

from wa.framework import pluginloader
from wa.framework.configuration.core import JobStatus


class Job(object):

    @property
    def id(self):
        return self.spec.id

    @property
    def output_name(self):
        return '{}-{}-{}'.format(self.id, self.spec.label, self.iteration)

    def __init__(self, spec, iteration, context):
        self.logger = logging.getLogger('job')
        self.spec = spec
        self.iteration = iteration
        self.context = context
        self.status = JobStatus.NEW
        self.workload = None
        self.output = None
        self.retries = 0

    def load(self, target, loader=pluginloader):
        self.logger.debug('Loading job {}'.format(self.id))
        self.workload = loader.get_workload(self.spec.workload_name,
                                            target,
                                            **self.spec.workload_parameters)
        self.workload.init_resources(self.context)
        self.workload.validate()
        self.status = JobStatus.LOADED

    def initialize(self, context):
        self.logger.info('Initializing job {}'.format(self.id))
        self.status = JobStatus.PENDING

    def configure_target(self, context):
        self.logger.info('Configuring target for job {}'.format(self.id))

    def setup(self, context):
        self.logger.info('Setting up job {}'.format(self.id))

    def run(self, context):
        self.logger.info('Running job {}'.format(self.id))

    def process_output(self, context):
        self.logger.info('Processing output for job {}'.format(self.id))

    def teardown(self, context):
        self.logger.info('Tearing down job {}'.format(self.id))

    def finalize(self, context):
        self.logger.info('Finalizing job {}'.format(self.id))

