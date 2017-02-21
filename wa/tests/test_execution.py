import os
import sys
import unittest
from StringIO import StringIO
from mock import Mock
from nose.tools import assert_true, assert_false, assert_equal

from wa.framework import signal
from wa.framework.agenda import Agenda
from wa.framework.run import RunnerJob
from wa.framework.execution import agenda_iterator

sys.path.insert(0, os.path.dirname(__file__))
from testutils import SignalWatcher


class TestAgendaIteration(unittest.TestCase):

    def setUp(self):
        agenda_text = """
        global:
            iterations: 2
        sections:
            - id: a
            - id: b
              workloads:
                - id: 1
                  name: bbench
        workloads:
            - id: 2
              name: dhrystone
            - id: 3
              name: coremark
              iterations: 1
        """
        agenda_file = StringIO(agenda_text)
        agenda_file.name = 'agenda'
        self.agenda = Agenda(agenda_file)

    def test_iteration_by_iteration(self):
        specs = ['{}-{}'.format(s.id, w.id) 
                 for _, s, w, _  
                 in agenda_iterator(self.agenda, 'by_iteration')]
        assert_equal(specs,
                     ['a-2', 'b-2', 'a-3', 'b-3', 'b-1', 'a-2', 'b-2', 'b-1'])

    def test_iteration_by_section(self):
        specs = ['{}-{}'.format(s.id, w.id) 
                 for _, s, w, _  
                 in agenda_iterator(self.agenda, 'by_section')]
        assert_equal(specs,
                     ['a-2', 'a-3', 'b-2', 'b-3', 'b-1', 'a-2', 'b-2', 'b-1'])

    def test_iteration_by_spec(self):
        specs = ['{}-{}'.format(s.id, w.id) 
                 for _, s, w, _  in 
                 agenda_iterator(self.agenda, 'by_spec')]
        assert_equal(specs,
                     ['a-2', 'a-2', 'a-3', 'b-2', 'b-2', 'b-3', 'b-1', 'b-1'])


class FakeWorkloadLoader(object):

    def get_workload(self, name, target, **params):
        workload = Mock()
        workload.name = name
        workload.target = target
        workload.parameters = params
        return workload


class WorkloadExecutionWatcher(SignalWatcher):

    signals = [
        signal.BEFORE_WORKLOAD_SETUP,
        signal.SUCCESSFUL_WORKLOAD_SETUP,
        signal.AFTER_WORKLOAD_SETUP,
        signal.BEFORE_WORKLOAD_EXECUTION,
        signal.SUCCESSFUL_WORKLOAD_EXECUTION,
        signal.AFTER_WORKLOAD_EXECUTION,
        signal.BEFORE_WORKLOAD_RESULT_UPDATE,
        signal.SUCCESSFUL_WORKLOAD_RESULT_UPDATE,
        signal.AFTER_WORKLOAD_RESULT_UPDATE,
        signal.BEFORE_WORKLOAD_TEARDOWN,
        signal.SUCCESSFUL_WORKLOAD_TEARDOWN,
        signal.AFTER_WORKLOAD_TEARDOWN,
    ]


class TestWorkloadExecution(unittest.TestCase):

    def setUp(self):
        params = {
            'target': Mock(),
            'context': Mock(),
            'loader': FakeWorkloadLoader(),
        }
        data = {
            'id': 'test',
            'workload': 'test',
            'label': None,
            'parameters': None,
        }
        self.job = RunnerJob('job1', 'execute-workload-job', params, data)
        self.workload = self.job.actor.workload
        self.watcher = WorkloadExecutionWatcher()

    def test_normal_flow(self):
        self.job.run()
        assert_true(self.workload.setup.called)
        assert_true(self.workload.run.called)
        assert_true(self.workload.update_result.called)
        assert_true(self.workload.teardown.called)
        self.watcher.assert_all_called()

    def test_failed_run(self):
        def bad(self):
            raise Exception()
        self.workload.run = bad
        try:
            self.job.run()
        except Exception:
            pass
        assert_true(self.workload.setup.called)
        assert_false(self.workload.update_result.called)
        assert_true(self.workload.teardown.called)

        assert_true(self.watcher.before_workload_setup.called)
        assert_true(self.watcher.successful_workload_setup.called)
        assert_true(self.watcher.after_workload_setup.called)
        assert_true(self.watcher.before_workload_execution.called)
        assert_false(self.watcher.successful_workload_execution.called)
        assert_true(self.watcher.after_workload_execution.called)
        assert_true(self.watcher.before_workload_result_update.called)
        assert_false(self.watcher.successful_workload_result_update.called)
        assert_true(self.watcher.after_workload_result_update.called)
        assert_true(self.watcher.before_workload_teardown.called)
        assert_true(self.watcher.successful_workload_teardown.called)
        assert_true(self.watcher.after_workload_teardown.called)

    def test_failed_setup(self):
        def bad(self):
            raise Exception()
        self.workload.setup = bad
        try:
            self.job.run()
        except Exception:
            pass
        assert_false(self.workload.run.called)
        assert_false(self.workload.update_result.called)
        assert_false(self.workload.teardown.called)

        assert_true(self.watcher.before_workload_setup.called)
        assert_false(self.watcher.successful_workload_setup.called)
        assert_true(self.watcher.after_workload_setup.called)
        assert_false(self.watcher.before_workload_execution.called)
        assert_false(self.watcher.successful_workload_execution.called)
        assert_false(self.watcher.after_workload_execution.called)
        assert_false(self.watcher.before_workload_result_update.called)
        assert_false(self.watcher.successful_workload_result_update.called)
        assert_false(self.watcher.after_workload_result_update.called)
        assert_false(self.watcher.before_workload_teardown.called)
        assert_false(self.watcher.successful_workload_teardown.called)
        assert_false(self.watcher.after_workload_teardown.called)
