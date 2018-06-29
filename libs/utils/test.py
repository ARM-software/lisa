# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2015, ARM Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import unittest
import logging

from bart.sched.SchedAssert import SchedAssert
from bart.sched.SchedMultiAssert import SchedMultiAssert
from devlib.utils.misc import memoized
import wrapt

from env import TestEnv
from executor import Executor
from trace import Trace


class LisaTest(unittest.TestCase):
    """
    A base class for LISA tests

    This class is intended to be subclassed in order to create automated tests
    for LISA. It sets up the TestEnv and Executor and provides convenience
    methods for making assertions on results.

    Subclasses should provide a test_conf to configure the TestEnv and an
    experiments_conf to configure the executor.

    Tests whose behaviour is dependent on target parameters, for example
    presence of cpufreq governors or number of CPUs, can override
    _getExperimentsConf to generate target-dependent experiments.

    Example users of this class can be found under LISA's tests/ directory.

    :ivar experiments: List of :class:`Experiment` s executed for the test. Only
                       available after :meth:`init` has been called.
    """

    test_conf = None
    """Override this with a dictionary or JSON path to configure the TestEnv"""

    experiments_conf = None
    """Override this with a dictionary or JSON path to configure the Executor"""

    permitted_fail_pct = 0
    """The percentage of iterations of each test that may be permitted to fail"""

    @classmethod
    def _getTestConf(cls):
        if cls.test_conf is None:
            raise NotImplementedError("Override `test_conf` attribute")
        return cls.test_conf

    @classmethod
    def _getExperimentsConf(cls, test_env):
        """
        Get the experiments_conf used to configure the Executor

        This method receives the initialized TestEnv as a parameter, so
        subclasses can override it to configure workloads or target confs in a
        manner dependent on the target. If not overridden, just returns the
        experiments_conf attribute.
        """
        if cls.experiments_conf is None:
            raise NotImplementedError("Override `experiments_conf` attribute")
        return cls.experiments_conf

    @classmethod
    def runExperiments(cls):
        """
        Set up logging and trigger running experiments
        """
        cls._log = logging.getLogger('LisaTest')

        cls._log.info('Setup tests execution engine...')
        test_env = TestEnv(test_conf=cls._getTestConf())

        experiments_conf = cls._getExperimentsConf(test_env)

        if ITERATIONS_FROM_CMDLINE:
            if 'iterations' in experiments_conf:
                cls.logger.warning(
                    "Command line overrides iteration count in "
                    "{}'s experiments_conf".format(cls.__name__))
            experiments_conf['iterations'] = ITERATIONS_FROM_CMDLINE

        cls.executor = Executor(test_env, experiments_conf)

        # Alias tests and workloads configurations
        cls.wloads = cls.executor._experiments_conf["wloads"]
        cls.confs = cls.executor._experiments_conf["confs"]

        # Alias executor objects to make less verbose tests code
        cls.te = cls.executor.te
        cls.target = cls.executor.target

        # Execute pre-experiments code defined by the test
        cls._experimentsInit()

        cls._log.info('Experiments execution...')
        cls.executor.run()

        cls.experiments = cls.executor.experiments

        # Execute post-experiments code defined by the test
        cls._experimentsFinalize()

    @classmethod
    def _experimentsInit(cls):
        """
        Code executed before running the experiments
        """

    @classmethod
    def _experimentsFinalize(cls):
        """
        Code executed after running the experiments
        """

    @memoized
    def get_sched_assert(self, experiment, task):
        """
        Return a SchedAssert over the task provided
        """
        return SchedAssert(
            self.get_trace(experiment).ftrace, self.te.topology, execname=task)

    @memoized
    def get_multi_assert(self, experiment, task_filter=""):
        """
        Return a SchedMultiAssert over the tasks whose names contain task_filter

        By default, this includes _all_ the tasks that were executed for the
        experiment.
        """
        tasks = experiment.wload.tasks.keys()
        return SchedMultiAssert(self.get_trace(experiment).ftrace,
                                self.te.topology,
                                [t for t in tasks if task_filter in t])

    def get_trace(self, experiment):
        if not hasattr(self, "__traces"):
            self.__traces = {}
        if experiment.out_dir in self.__traces:
            return self.__traces[experiment.out_dir]

        if ('ftrace' not in experiment.conf['flags']
            or 'ftrace' not in self.test_conf):
            raise ValueError(
                'Tracing not enabled. If this test needs a trace, add "ftrace" '
                'to your test/experiment configuration flags')

        events = self.test_conf['ftrace']['events']
        trace = Trace(experiment.out_dir, events, self.te.platform)

        self.__traces[experiment.out_dir] = trace
        return trace

    def get_start_time(self, experiment):
        """
        Get the time at which the experiment workload began executing
        """
        start_times_dict = self.get_multi_assert(experiment).getStartTime()
        return min([t["starttime"] for t in start_times_dict.itervalues()])

    def get_task_start_time(self, experiment, task):
        """
        Get the time at which the input task of the experiment workload began
        executing
        """
        start_times_dict = self.get_multi_assert(experiment, task_filter=task).getStartTime()
        return min([t["starttime"] for t in start_times_dict.itervalues()])

    def get_end_time(self, experiment):
        """
        Get the time at which the experiment workload finished executing
        """
        end_times_dict = self.get_multi_assert(experiment).getEndTime()
        return max([t["endtime"] for t in end_times_dict.itervalues()])

    def get_window(self, experiment):
        return (self.get_start_time(experiment), self.get_end_time(experiment))

    def get_end_times(self, experiment):
        """
        Get the time at which each task in the workload finished

        Returned as a dict; {"task_name": finish_time, ...}
        """

        end_times = {}
        ftrace = self.get_trace(experiment).ftrace
        for task in experiment.wload.tasks.keys():
            sched_assert = SchedAssert(ftrace, self.te.topology, execname=task)
            end_times[task] = sched_assert.getEndTime()

        return end_times

    def _dummy_method(self):
        pass

    # In the Python unittest framework you instantiate TestCase objects passing
    # the name of a test method that is going to be run to make assertions. We
    # run our tests using nosetests, which automatically discovers these
    # methods. However we also want to be able to instantiate LisaTest objects
    # in notebooks without the inconvenience of having to provide a methodName,
    # since we won't need any assertions. So we'll override __init__ with a
    # default dummy test method that does nothing.
    def __init__(self, methodName='_dummy_method', *args, **kwargs):
        super(LisaTest, self).__init__(methodName, *args, **kwargs)

@wrapt.decorator
def experiment_test(wrapped_test, instance, args, kwargs):
    """
    Convert a LisaTest test method to be automatically called for each experiment

    The method will be passed the experiment object and a list of the names of
    tasks that were run as the experiment's workload.
    """
    failures = {}
    for experiment in instance.executor.experiments:
        tasks = experiment.wload.tasks.keys()
        try:
            wrapped_test(experiment, tasks, *args, **kwargs)
        except AssertionError as e:
            trace_relpath = os.path.join(experiment.out_dir, "trace.dat")
            add_msg = "Check trace file: " + os.path.abspath(trace_relpath)
            msg = str(e) + "\n\t" +  add_msg

            test_key = (experiment.wload_name, experiment.conf['tag'])
            failures[test_key] = failures.get(test_key, []) + [msg]

    for fails in failures.itervalues():
        iterations = instance.executor.iterations
        fail_pct = 100. * len(fails) / iterations

        msg = "{} failures from {} iteration(s):\n{}".format(
            len(fails), iterations, '\n'.join(fails))
        if fail_pct > instance.permitted_fail_pct:
            raise AssertionError(msg)
        else:
            instance._log.warning(msg)
            instance._log.warning(
                'ALLOWING due to permitted_fail_pct={}'.format(
                    instance.permitted_fail_pct))


# Prevent nosetests from running experiment_test directly as a test case
experiment_test.__test__ = False

# Allow the user to override the iterations setting from the command
# line. Nosetests does not support this kind of thing, so we use an
# evil hack: the lisa-test shell function takes an --iterations
# argument and exports an environment variable. If the test itself
# specifies an iterations count, we'll later print a warning and
# override it. We do this here in the root scope, rather than in
# runExperiments, so that if the value is invalid we print the error
# immediately instead of going ahead with target setup etc.
try:
    iterations = os.getenv('LISA_TEST_ITERATIONS')
    # Empty string or 0 will be replaced by 0, otherwise converted to int
    ITERATIONS_FROM_CMDLINE = int(iterations) if iterations else 0

    if ITERATIONS_FROM_CMDLINE < 0:
        raise ValueError('Cannot be negative')
except ValueError as e:
    raise ValueError("Couldn't read iterations count: {}".format(e))

# vim :set tabstop=4 shiftwidth=4 expandtab textwidth=80
