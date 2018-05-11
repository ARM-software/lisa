import logging
from copy import copy
from datetime import datetime

from wa.framework import pluginloader, signal, instrument
from wa.framework.configuration.core import Status
from wa.utils.log import indentcontext

# Because of use of Enum (dynamic attrs)
# pylint: disable=no-member

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

    @property
    def has_been_initialized(self):
        return self._has_been_initialized

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
        self._has_been_initialized = False
        self._status = Status.NEW

    def load(self, target, loader=pluginloader):
        self.logger.info('Loading job {}'.format(self))
        if self.iteration == 1:
            self.workload = loader.get_workload(self.spec.workload_name,
                                                target,
                                                **self.spec.workload_parameters)
            self.workload.init_resources(self.context)
            self.workload.validate()
            self._workload_cache[self.id] = self.workload
        else:
            self.workload = self._workload_cache[self.id]

    def set_output(self, output):
        output.classifiers = copy(self.classifiers)
        self.output = output

    def initialize(self, context):
        self.logger.info('Initializing job {}'.format(self))
        with indentcontext():
            with signal.wrap('WORKLOAD_INITIALIZED', self, context):
                self.workload.logger.context = context
                self.workload.initialize(context)
            self.set_status(Status.PENDING)
            self._has_been_initialized = True
            context.update_job_state(self)

    def configure_augmentations(self, context, pm):
        self.logger.info('Configuring augmentations')
        with indentcontext():
            instruments_to_enable = set()
            output_processors_to_enable = set()
            enabled_instruments = set(i.name for i in instrument.get_enabled())
            enabled_output_processors = set(p.name for p in pm.get_enabled())

            for augmentation in self.spec.augmentations.values():
                augmentation_cls = context.cm.plugin_cache.get_plugin_class(augmentation)
                if augmentation_cls.kind == 'instrument':
                    instruments_to_enable.add(augmentation)
                elif augmentation_cls.kind == 'output_processor':
                    output_processors_to_enable.add(augmentation)

            # Disable unrequired instruments
            for instrument_name in enabled_instruments.difference(instruments_to_enable):
                instrument.disable(instrument_name)
            # Enable additional instruments
            for instrument_name in instruments_to_enable.difference(enabled_instruments):
                instrument.enable(instrument_name)

            # Disable unrequired output_processors
            for processor in enabled_output_processors.difference(output_processors_to_enable):
                pm.disable(processor)
            # Enable additional output_processors
            for processor in output_processors_to_enable.difference(enabled_output_processors):
                pm.enable(processor)

    def configure_target(self, context):
        self.logger.info('Configuring target for job {}'.format(self))
        context.tm.commit_runtime_parameters(self.spec.runtime_parameters)

    def setup(self, context):
        self.logger.info('Setting up job {}'.format(self))
        with signal.wrap('WORKLOAD_SETUP', self, context):
            self.workload.setup(context)

    def run(self, context):
        self.logger.info('Running job {}'.format(self))
        with signal.wrap('WORKLOAD_EXECUTION', self, context):
            start_time = datetime.utcnow()
            try:
                self.workload.run(context)
            finally:
                self.run_time = datetime.utcnow() - start_time

    def process_output(self, context):
        if not context.tm.is_responsive:
            self.logger.info('Target unresponsive; not processing job output.')
            return
        self.logger.info('Processing output for job {}'.format(self))
        with indentcontext():
            if self.status != Status.FAILED:
                with signal.wrap('WORKLOAD_RESULT_EXTRACTION', self, context):
                    self.workload.extract_results(context)
                    context.extract_results()
                with signal.wrap('WORKLOAD_OUTPUT_UPDATE', self, context):
                    self.workload.update_output(context)

    def teardown(self, context):
        if not context.tm.is_responsive:
            self.logger.info('Target unresponsive; not tearing down.')
            return
        self.logger.info('Tearing down job {}'.format(self))
        with indentcontext():
            with signal.wrap('WORKLOAD_TEARDOWN', self, context):
                self.workload.teardown(context)

    def finalize(self, context):
        if not self._has_been_initialized:
            return
        if not context.tm.is_responsive:
            self.logger.info('Target unresponsive; not finalizing.')
            return
        self.logger.info('Finalizing job {} '.format(self))
        with indentcontext():
            with signal.wrap('WORKLOAD_FINALIZED', self, context):
                self.workload.finalize(context)

    def set_status(self, status, force=False):
        status = Status(status)
        if force or self.status < status:
            self.status = status

    def __str__(self):
        return '{} ({}) [{}]'.format(self.id, self.label, self.iteration)

    def __repr__(self):
        return 'Job({})'.format(self)
