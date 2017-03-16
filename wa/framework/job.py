import logging

from wa.framework import pluginloader
from wa.framework.configuration.core import JobStatus


class Job(object):

    @property
    def id(self):
        return self.spec.id

    @property
    def label(self):
        return self.spec.label

    @property
    def classifiers(self):
        return self.spec.classifiers

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

    def initialize(self, context):
        self.logger.info('Initializing job {}'.format(self.id))
        self.workload.initialize(context)
        self.status = JobStatus.PENDING
        context.update_job_state(self)

    def configure_target(self, context):
        self.logger.info('Configuring target for job {}'.format(self.id))

    def setup(self, context):
        self.logger.info('Setting up job {}'.format(self.id))
        self.workload.setup(context)

    def run(self, context):
        self.logger.info('Running job {}'.format(self.id))
        self.workload.run(context)

    def process_output(self, context):
        self.logger.info('Processing output for job {}'.format(self.id))
        self.workload.update_result(context)

    def teardown(self, context):
        self.logger.info('Tearing down job {}'.format(self.id))
        self.workload.teardown(context)

    def finalize(self, context):
        self.logger.info('Finalizing job {}'.format(self.id))
        self.workload.finalize(context)

