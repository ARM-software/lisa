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

# pylint: disable=no-member

"""
This module defines the classes used to handle result
processing inside Workload Automation. There will be a
:class:`wlauto.core.workload.WorkloadResult` object generated for
every workload iteration executed. This object will have a list of
:class:`wlauto.core.workload.WorkloadMetric` objects. This list will be
populated by the workload itself and may also be updated by instrumentation
(e.g. to add power measurements).  Once the result object has been fully
populated, it will be passed into the ``process_iteration_result`` method of
:class:`ResultProcessor`. Once the entire run has completed, a list containing
result objects from all iterations will be passed into ``process_results``
method of :class`ResultProcessor`.

Which result processors will be active is defined by the ``result_processors``
list in the ``~/.workload_automation/config.py``. Only the result_processors
who's names appear in this list will be used.

A :class:`ResultsManager`  keeps track of active results processors.

"""
import logging
import traceback
from copy import copy
from contextlib import contextmanager
from datetime import datetime

from wlauto.core.plugin import Plugin
from wlauto.exceptions import WAError
from wlauto.utils.types import numeric
from wlauto.utils.misc import enum_metaclass, merge_dicts


class ResultManager(object):
    """
    Keeps track of result processors and passes on the results onto the individual processors.

    """

    def __init__(self):
        self.logger = logging.getLogger('ResultsManager')
        self.processors = []
        self._bad = []

    def install(self, processor):
        self.logger.debug('Installing results processor %s', processor.name)
        self.processors.append(processor)

    def uninstall(self, processor):
        if processor in self.processors:
            self.logger.debug('Uninstalling results processor %s', processor.name)
            self.processors.remove(processor)
        else:
            self.logger.warning('Attempting to uninstall results processor %s, which is not installed.',
                                processor.name)

    def initialize(self, context):
        # Errors aren't handled at this stage, because this gets executed
        # before workload execution starts and we just want to propagte them
        # and terminate (so that error can be corrected and WA restarted).
        for processor in self.processors:
            processor.initialize(context)

    def add_result(self, result, context):
        with self._manage_processors(context):
            for processor in self.processors:
                with self._handle_errors(processor):
                    processor.process_iteration_result(result, context)
            for processor in self.processors:
                with self._handle_errors(processor):
                    processor.export_iteration_result(result, context)

    def process_run_result(self, result, context):
        with self._manage_processors(context):
            for processor in self.processors:
                with self._handle_errors(processor):
                    processor.process_run_result(result, context)
            for processor in self.processors:
                with self._handle_errors(processor):
                    processor.export_run_result(result, context)

    def finalize(self, context):
        with self._manage_processors(context):
            for processor in self.processors:
                with self._handle_errors(processor):
                    processor.finalize(context)

    def validate(self):
        for processor in self.processors:
            processor.validate()

    @contextmanager
    def _manage_processors(self, context, finalize_bad=True):
        yield
        for processor in self._bad:
            if finalize_bad:
                processor.finalize(context)
            self.uninstall(processor)
        self._bad = []

    @contextmanager
    def _handle_errors(self, processor):
        try:
            yield
        except KeyboardInterrupt, e:
            raise e
        except WAError, we:
            self.logger.error('"{}" result processor has encountered an error'.format(processor.name))
            self.logger.error('{}("{}")'.format(we.__class__.__name__, we.message))
            self._bad.append(processor)
        except Exception, e:  # pylint: disable=W0703
            self.logger.error('"{}" result processor has encountered an error'.format(processor.name))
            self.logger.error('{}("{}")'.format(e.__class__.__name__, e))
            self.logger.error(traceback.format_exc())
            self._bad.append(processor)


class ResultProcessor(Plugin):
    """
    Base class for result processors. Defines an interface that should be implemented
    by the subclasses. A result processor can be used to do any kind of post-processing
    of the results, from writing them out to a file, to uploading them to a database,
    performing calculations, generating plots, etc.

    """
    kind = "result_processor"
    def initialize(self, context):
        pass

    def process_iteration_result(self, result, context):
        pass

    def export_iteration_result(self, result, context):
        pass

    def process_run_result(self, result, context):
        pass

    def export_run_result(self, result, context):
        pass

    def finalize(self, context):
        pass


class RunResult(object):
    """
    Contains overall results for a run.

    """

    __metaclass__ = enum_metaclass('values', return_name=True)

    values = [
        'OK',
        'OKISH',
        'PARTIAL',
        'FAILED',
        'UNKNOWN',
    ]

    @property
    def status(self):
        if not self.iteration_results or all([s.status == IterationResult.FAILED for s in self.iteration_results]):
            return self.FAILED
        elif any([s.status == IterationResult.FAILED for s in self.iteration_results]):
            return self.PARTIAL
        elif any([s.status == IterationResult.ABORTED for s in self.iteration_results]):
            return self.PARTIAL
        elif (any([s.status == IterationResult.PARTIAL for s in self.iteration_results]) or
                self.non_iteration_errors):
            return self.OKISH
        elif all([s.status == IterationResult.OK for s in self.iteration_results]):
            return self.OK
        else:
            return self.UNKNOWN  # should never happen

    def __init__(self, run_info, output_directory=None):
        self.info = run_info
        self.iteration_results = []
        self.artifacts = []
        self.events = []
        self.non_iteration_errors = False
        self.output_directory = output_directory


class RunEvent(object):
    """
    An event that occured during a run.

    """
    def __init__(self, message):
        self.timestamp = datetime.utcnow()
        self.message = message

    def to_dict(self):
        return copy(self.__dict__)

    def __str__(self):
        return '{} {}'.format(self.timestamp, self.message)

    __repr__ = __str__


class IterationResult(object):
    """
    Contains the result of running a single iteration of a workload. It is the
    responsibility of a workload to instantiate a IterationResult, populate it,
    and return it form its get_result() method.

    Status explanations:

       :NOT_STARTED: This iteration has not yet started.
       :RUNNING: This iteration is currently running and no errors have been detected.
       :OK: This iteration has completed and no errors have been detected
       :PARTIAL: One or more instruments have failed (the iteration may still be running).
       :FAILED: The workload itself has failed.
       :ABORTED: The user interupted the workload
       :SKIPPED: The iteration was skipped due to a previous failure

    """

    __metaclass__ = enum_metaclass('values', return_name=True)

    values = [
        'NOT_STARTED',
        'RUNNING',

        'OK',
        'NONCRITICAL',
        'PARTIAL',
        'FAILED',
        'ABORTED',
        'SKIPPED',
    ]

    def __init__(self, spec):
        self.spec = spec
        self.id = spec.id
        self.workload = spec.workload
        self.classifiers = copy(spec.classifiers)
        self.iteration = None
        self.status = self.NOT_STARTED
        self.output_directory = None
        self.events = []
        self.metrics = []
        self.artifacts = []

    def add_metric(self, name, value, units=None, lower_is_better=False, classifiers=None):
        classifiers = merge_dicts(self.classifiers, classifiers or {},
                                  list_duplicates='last', should_normalize=False)
        self.metrics.append(Metric(name, value, units, lower_is_better, classifiers))

    def has_metric(self, name):
        for metric in self.metrics:
            if metric.name == name:
                return True
        return False

    def add_event(self, message):
        self.events.append(RunEvent(message))

    def to_dict(self):
        d = copy(self.__dict__)
        d['events'] = [e.to_dict() for e in self.events]
        return d

    def __iter__(self):
        return iter(self.metrics)

    def __getitem__(self, name):
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError('Metric {} not found.'.format(name))


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

    def __init__(self, name, value, units=None, lower_is_better=False, classifiers=None):
        self.name = name
        self.value = numeric(value)
        self.units = units
        self.lower_is_better = lower_is_better
        self.classifiers = classifiers or {}

    def to_dict(self):
        return self.__dict__

    def __str__(self):
        result = '{}: {}'.format(self.name, self.value)
        if self.units:
            result += ' ' + self.units
        result += ' ({})'.format('-' if self.lower_is_better else '+')
        return '<{}>'.format(result)

    __repr__ = __str__
