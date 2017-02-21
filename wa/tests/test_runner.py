import os
import sys
import shutil
import tempfile
import unittest

from mock import Mock
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal

from wa.framework import pluginloader
from wa.framework.output import RunOutput
from wa.framework.run import Runner, RunnerJob, runmethod, reset_runmethods
from wa.utils.serializer import json



class RunnerTest(unittest.TestCase):

    def setUp(self):
        self.output = RunOutput(tempfile.mktemp())
        self.output.initialize()

    def tearDown(self):
        shutil.rmtree(self.output.output_directory)

    def test_run_init(self):
        runner = Runner(self.output)
        runner.initialize()
        runner.finalize()
        assert_true(runner.info.name)
        assert_true(runner.info.start_time)
        assert_true(runner.info.end_time)
        assert_almost_equal(runner.info.duration,
                            runner.info.end_time -
                            runner.info.start_time)

    def test_normal_run(self):
        runner = Runner(self.output)
        runner.add_job(1, Mock())
        runner.add_job(2, Mock())
        runner.initialize()
        runner.run()
        runner.finalize()
        assert_equal(len(runner.completed_jobs), 2)
