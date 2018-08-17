from __future__ import unicode_literals
#    Copyright 2017 ARM Limited, Google
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

import os
import sys

import utils_tests
import trappy

sys.path.append(os.path.join(utils_tests.TESTS_DIRECTORY, "..", "trappy"))

class TestFilesystem(utils_tests.SetupDirectory):
    def __init__(self, *args, **kwargs):
        super(TestFilesystem, self).__init__(
             [("trace_filesystem.txt", "trace_filesystem.txt"),],
             *args,
             **kwargs)

    #ext4 tests
    def test_filesystem_ext_da_write_begin_can_be_parsed(self):
        """TestFilesystem: test that ext4_da_write_begin events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['ext4_da_write_begin'])
        df = trace.ext4_da_write_begin.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "pos", "len", "flags"]))

    def test_filesystem_ext_da_write_end_can_be_parsed(self):
        """TestFilesystem: test that ext4_da_write_end events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['ext4_da_write_end'])
        df = trace.ext4_da_write_end.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "pos", "len", "copied"]))

    def test_filesystem_ext_sync_file_enter_can_be_parsed(self):
        """TestFilesystem: test that ext4_sync_file_enter events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['ext4_sync_file_enter'])
        df = trace.ext4_sync_file_enter.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "parent", "datasync"]))

    def test_filesystem_ext_sync_file_exit_can_be_parsed(self):
        """TestFilesystem: test that ext4_sync_file_exit events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['ext4_sync_file_exit'])
        df = trace.ext4_sync_file_exit.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "ret"]))

    #f2fs tests
    def test_filesystem_f2fs_write_begin_can_be_parsed(self):
        """TestFilesystem: test that f2fs_write_begin events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['f2fs_write_begin'])
        df = trace.f2fs_write_begin.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "pos", "len", "flags"]))

    def test_filesystem_f2fs_write_end_can_be_parsed(self):
        """TestFilesystem: test that f2fs_write_end events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['f2fs_write_end'])
        df = trace.f2fs_write_end.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "pos", "len", "copied"]))

    def test_filesystem_f2fs_sync_file_enter_can_be_parsed(self):
        """TestFilesystem: test that f2fs_sync_file_enter events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['f2fs_sync_file_enter'])
        df = trace.f2fs_sync_file_enter.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev",
                                 "inode", "pino", "i_mode", "i_size", "i_nlink", "i_blocks", "i_advise"]))

    def test_filesystem_f2fs_sync_file_exit_can_be_parsed(self):
        """TestFilesystem: test that f2fs_sync_file_exit events can be parsed"""
        trace = trappy.FTrace("trace_filesystem.txt", events=['f2fs_sync_file_exit'])
        df = trace.f2fs_sync_file_exit.data_frame
        self.assertSetEqual(set(df.columns),
                            set(["__comm", "__cpu", "__line", "__pid", "dev", "inode", "checkpoint", "datasync", "ret"]))
