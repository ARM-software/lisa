import os
import logging
import shutil
import random
from copy import copy
from collections import OrderedDict, defaultdict

from wa.framework import pluginloader, signal, log
from wa.framework.run import Runner, RunnerJob
from wa.framework.output import RunOutput
from wa.framework.actor import JobActor
from wa.framework.resource import ResourceResolver
from wa.framework.exception import ConfigError, NotFoundError
from wa.framework.configuration import ConfigurationPoint, PluginConfiguration, WA_CONFIGURATION
from wa.utils.serializer import read_pod
from wa.utils.misc import ensure_directory_exists as _d, Namespace
from wa.utils.types import list_of, identifier, caseless_string


__all__ = [
    'Executor',
    'ExecutionOutput',
    'ExecutionwContext',
    'ExecuteWorkloadContainerActor',
    'ExecuteWorkloadJobActor',
]


class Executor(object):

    def __init__(self, output):
        self.output = output
        self.config = ExecutionRunConfiguration()
        self.agenda_string =  None
        self.agenda = None
        self.jobs = None
        self.container = None
        self.target = None

    def load_config(self, filepath):
        self.config.update(filepath)

    def load_agenda(self, agenda_string):
        if self.agenda:
            raise RuntimeError('Only one agenda may be loaded per run.')
        self.agenda_string = agenda_string
        if os.path.isfile(agenda_string):
            self.logger.debug('Loading agenda from {}'.format(agenda_string))
            self.agenda = Agenda(agenda_string)
            shutil.copy(agenda_string, self.output.config_directory)
        else:
            self.logger.debug('"{}" is not a file; assuming workload name.'.format(agenda_string))
            self.agenda = Agenda()
            self.agenda.add_workload_entry(agenda_string)

    def disable_instrument(self, name):
        if not self.agenda:
            raise RuntimeError('initialize() must be invoked before disable_instrument()')
        self.agenda.config['instrumentation'].append('~{}'.format(itd))

    def initialize(self):
        if not self.agenda:
            raise RuntimeError('No agenda has been loaded.')
        self.config.update(self.agenda.config)
        self.config.consolidate()
        self._initialize_target()
        self._initialize_job_config()

    def execute(self, selectors=None):
        pass

    def finalize(self):
        pass

    def _initialize_target(self):
        pass

    def _initialize_job_config(self):
        self.agenda.expand(self.target)
        for tup in agenda_iterator(self.agenda, self.config.execution_order):
            glob, sect, workload, iter_number = tup


def agenda_iterator(agenda, order):
    """
    Iterates over all job components in an agenda, yielding tuples in the form ::

        (global_entry, section_entry, workload_entry, iteration_number)

    Which fully define the job to be crated. The order in which these tuples are 
    yielded is determined by the ``order`` parameter which may be one of the following
    values:

    ``"by_iteration"`` 
      The first iteration of each workload spec is executed one after the other,
      so all workloads are executed before proceeding on to the second iteration.
      E.g. A1 B1 C1 A2 C2 A3. This is the default if no order is explicitly specified.
 
      In case of multiple sections, this will spread them out, such that specs
      from the same section are further part. E.g. given sections X and Y, global
      specs A and B, and two iterations, this will run ::
 
                      X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2
 
    ``"by_section"`` 
      Same  as ``"by_iteration"``, however this will group specs from the same
      section together, so given sections X and Y, global specs A and B, and two iterations, 
      this will run ::
 
              X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2
 
    ``"by_spec"``
      All iterations of the first spec are executed before moving on to the next
      spec. E.g. A1 A2 A3 B1 C1 C2. 
 
    ``"random"``
      Execution order is entirely random.

    """
    # TODO: this would be a place to perform section expansions.
    #       (e.g. sweeps, cross-products, etc).

    global_iterations = agenda.global_.number_of_iterations
    all_iterations = [global_iterations]
    all_iterations.extend([s.number_of_iterations for s in agenda.sections])
    all_iterations.extend([w.number_of_iterations for w in agenda.workloads])
    max_iterations = max(all_iterations)

    if order == 'by_spec':
        if agenda.sections:
            for section in agenda.sections:
                section_iterations = section.number_of_iterations or global_iterations
                for workload in agenda.workloads + section.workloads:
                    workload_iterations =  workload.number_of_iterations or section_iterations
                    for i in xrange(workload_iterations):
                        yield agenda.global_, section, workload, i
        else:  # not sections
            for workload in agenda.workloads:
                workload_iterations =  workload.number_of_iterations or global_iterations
                for i in xrange(workload_iterations):
                    yield agenda.global_, None, workload, i
    elif order == 'by_section':
        for i in xrange(max_iterations):
            if agenda.sections:
                for section in agenda.sections:
                    section_iterations = section.number_of_iterations or global_iterations
                    for workload in agenda.workloads + section.workloads:
                        workload_iterations =  workload.number_of_iterations or section_iterations
                        if i < workload_iterations:
                            yield agenda.global_, section, workload, i
            else:  # not sections
                for workload in agenda.workloads:
                    workload_iterations =  workload.number_of_iterations or global_iterations
                    if i < workload_iterations:
                        yield agenda.global_, None, workload, i
    elif order == 'by_iteration':
        for i in xrange(max_iterations):
            if agenda.sections:
                for workload in agenda.workloads:
                    for section in agenda.sections:
                        section_iterations = section.number_of_iterations or global_iterations
                        workload_iterations =  workload.number_of_iterations or section_iterations or global_iterations
                        if i < workload_iterations:
                            yield agenda.global_, section, workload, i
                # Now do the section-specific workloads
                for section in agenda.sections:
                    section_iterations = section.number_of_iterations or global_iterations
                    for workload in section.workloads:
                        workload_iterations =  workload.number_of_iterations or section_iterations or global_iterations
                        if i < workload_iterations:
                            yield agenda.global_, section, workload, i
            else:  # not sections
                for workload in agenda.workloads:
                    workload_iterations =  workload.number_of_iterations or global_iterations
                    if i < workload_iterations:
                        yield agenda.global_, None, workload, i
    elif order == 'random':
        tuples = list(agenda_iterator(data, order='by_section'))
        random.shuffle(tuples)
        for t in tuples:
            yield t
    else:
        raise ValueError('Invalid order: "{}"'.format(order))



class RebootPolicy(object):
    """
    Represents the reboot policy for the execution -- at what points the device
    should be rebooted. This, in turn, is controlled by the policy value that is
    passed in on construction and would typically be read from the user's settings.
    Valid policy values are:

    :never: The device will never be rebooted.
    :as_needed: Only reboot the device if it becomes unresponsive, or needs to be flashed, etc.
    :initial: The device will be rebooted when the execution first starts, just before
              executing the first workload spec.
    :each_spec: The device will be rebooted before running a new workload spec.
    :each_iteration: The device will be rebooted before each new iteration.

    """

    valid_policies = ['never', 'as_needed', 'initial', 'each_spec', 'each_iteration']

    def __init__(self, policy):
        policy = policy.strip().lower().replace(' ', '_')
        if policy not in self.valid_policies:
            message = 'Invalid reboot policy {}; must be one of {}'.format(policy, ', '.join(self.valid_policies))
            raise ConfigError(message)
        self.policy = policy

    @property
    def can_reboot(self):
        return self.policy != 'never'

    @property
    def perform_initial_boot(self):
        return self.policy not in ['never', 'as_needed']

    @property
    def reboot_on_each_spec(self):
        return self.policy in ['each_spec', 'each_iteration']

    @property
    def reboot_on_each_iteration(self):
        return self.policy == 'each_iteration'

    def __str__(self):
        return self.policy

    __repr__ = __str__

    def __cmp__(self, other):
        if isinstance(other, RebootPolicy):
            return cmp(self.policy, other.policy)
        else:
            return cmp(self.policy, other)


class RuntimeParameterSetter(object):
    """
    Manages runtime parameter state during execution.

    """

    @property
    def target(self):
        return self.target_assistant.target

    def __init__(self, target_assistant):
        self.target_assistant = target_assistant
        self.to_set = defaultdict(list) # name --> list of values 
        self.last_set = {}
        self.to_unset = defaultdict(int) # name --> count

    def validate(self, params):
        self.target_assistant.validate_runtime_parameters(params)

    def mark_set(self, params):
        for name, value in params.iteritems():
            self.to_set[name].append(value)
            
    def mark_unset(self, params):
        for name in params.iterkeys():
            self.to_unset[name] += 1

    def inact_set(self):
        self.target_assistant.clear_parameters()
        for name in self.to_set:
            self._set_if_necessary(name)
        self.target_assitant.set_parameters()
        
    def inact_unset(self):
        self.target_assistant.clear_parameters()
        for name, count in self.to_unset.iteritems():
            while count:
                self.to_set[name].pop()
                count -= 1
            self._set_if_necessary(name)
        self.target_assitant.set_parameters()

    def _set_if_necessary(self, name):
        if not self.to_set[name]:
            return
        new_value = self.to_set[name][-1]
        prev_value = self.last_set.get(name)
        if new_value != prev_value:
            self.target_assistant.add_paramter(name, new_value)
            self.last_set[name] = new_value


class WorkloadExecutionConfig(object):

    @staticmethod
    def from_pod(pod):
        return WorkloadExecutionConfig(**pod)

    def __init__(self, workload_name, workload_parameters=None,
                 runtime_parameters=None, components=None, 
                 assumptions=None):
        self.workload_name = workload_name or None
        self.workload_parameters = workload_parameters or {}
        self.runtime_parameters = runtime_parameters or {}
        self.components = components or {}
        self.assumpations = assumptions or {}

    def to_pod(self):
        return copy(self.__dict__)


class WorkloadExecutionActor(JobActor):

    def __init__(self, target, config, loader=pluginloader):
        self.target = target
        self.config = config
        self.logger = logging.getLogger('exec')
        self.context = None
        self.workload = loader.get_workload(config.workload_name, target, 
                                            **config.workload_parameters)
    def get_config(self):
        return self.config.to_pod()

    def initialize(self, context):
        self.context = context
        self.workload.init_resources(self.context)
        self.workload.validate()
        self.workload.initialize(self.context)

    def run(self):
        if not self.workload:
            self.logger.warning('Failed to initialize workload; skipping execution')
            return
        self.pre_run()
        self.logger.info('Setting up workload')
        with signal.wrap('WORKLOAD_SETUP'):
            self.workload.setup(self.context)
        try:
            error = None
            self.logger.info('Executing workload')
            try:
                with signal.wrap('WORKLOAD_EXECUTION'):
                    self.workload.run(self.context)
            except Exception as e:
                log.log_error(e, self.logger)
                error = e

            self.logger.info('Processing execution results')
            with signal.wrap('WORKLOAD_RESULT_UPDATE'):
                if not error:
                    self.workload.update_result(self.context)
                else:
                    self.logger.info('Workload execution failed; not extracting workload results.')
                    raise error
        finally:
            if self.target.check_responsive():
                self.logger.info('Tearing down workload')
                with signal.wrap('WORKLOAD_TEARDOWN'):
                    self.workload.teardown(self.context)
            self.post_run()

    def finalize(self):
        self.workload.finalize(self.context)

    def pre_run(self):
        # TODO: enable components, etc
        pass

    def post_run(self):
        pass
