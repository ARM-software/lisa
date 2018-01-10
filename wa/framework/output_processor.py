import logging

from wa.framework import pluginloader
from wa.framework.exception import ConfigError
from wa.framework.instruments import is_installed
from wa.framework.plugin import Plugin
from wa.utils.log import log_error, indent, dedent


class OutputProcessor(Plugin):

    kind = 'output_processor'
    requires = []

    def validate(self):
        super(OutputProcessor, self).validate()
        for instrument in self.requires:
            if not is_installed(instrument):
                msg = 'Instrument "{}" is required by {}, but is not installed.'
                raise ConfigError(msg.format(instrument, self.name))

    def initialize(self):
        pass

    def finalize(self):
        pass


class ProcessorManager(object):

    def __init__(self, loader=pluginloader):
        self.loader = loader
        self.logger = logging.getLogger('processor')
        self.processors = []

    def install(self, processor, context):
        if not isinstance(processor, OutputProcessor):
            processor = self.loader.get_output_processor(processor)
        self.logger.debug('Installing {}'.format(processor.name))
        processor.logger.context = context
        self.processors.append(processor)

    def validate(self):
        for proc in self.processors:
            proc.validate()

    def initialize(self):
        for proc in self.processors:
            proc.initialize()

    def finalize(self):
        for proc in self.processors:
            proc.finalize()

    def process_job_output(self, context):
        self.do_for_each_proc('process_job_output', 'processing using "{}"',
                              context.job_output, context.target_info,
                              context.run_output)

    def export_job_output(self, context):
        self.do_for_each_proc('export_job_output', 'Exporting using "{}"',
                              context.job_output, context.target_info,
                              context.run_output)

    def process_run_output(self, context):
        self.do_for_each_proc('process_run_output', 'Processing using "{}"',
                              context.run_output, context.target_info)

    def export_run_output(self, context):
        self.do_for_each_proc('export_run_output', 'Exporting using "{}"',
                              context.run_output, context.target_info)

    def do_for_each_proc(self, method_name, message, *args):
        try:
            indent()
            for proc in self.processors:
                proc_func = getattr(proc, method_name, None)
                if proc_func is None:
                    continue
                try:
                    self.logger.info(message.format(proc.name))
                    proc_func(*args)
                except Exception as e:
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    log_error(e, self.logger)
        finally:
            dedent()
