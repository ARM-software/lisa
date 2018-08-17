#    Copyright 2017 Google, ARM Limited
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


"""
Definitions of filesystem (ext4) trace parsers
registered by the FTrace class
"""
from __future__ import unicode_literals

from builtins import zip
from trappy.base import Base
from trappy.dynamic import register_ftrace_parser, register_dynamic_ftrace

class FilesystemExt4Base(Base):
    def generate_data_dict(self, data_str):
        #filesystem traces are space delimited in the form:
        #fieldA valueA fieldB valueB ...
        data = data_str.split(' ')
        return list(zip(data[0::2], data[1::2]))

    def finalize_object(self):
        self.data_frame.rename(columns={'ino':'inode'}, inplace=True)


class FilesystemExt4DaWriteBegin(FilesystemExt4Base):
    """Corresponds to Linux kernel trace event ext4_da_write_begin"""

    unique_word = "ext4_da_write_begin:"
    """The unique word that will be matched in a trace line"""


register_ftrace_parser(FilesystemExt4DaWriteBegin)

class FilesystemExt4DaWriteEnd(FilesystemExt4Base):
    """Corresponds to Linux kernel trace event ext4_da_write_end"""

    unique_word = "ext4_da_write_end:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemExt4DaWriteEnd)

class FilesystemExt4SyncFileEnter(FilesystemExt4Base):
    """Corresponds to Linux kernel trace event ext4_sync_file_enter"""

    unique_word = "ext4_sync_file_enter:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemExt4SyncFileEnter)

class FilesystemExt4SyncFileExit(FilesystemExt4Base):
    """Corresponds to Linux kernel trace event ext4_sync_file_exit"""

    unique_word = "ext4_sync_file_exit:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemExt4SyncFileExit)

class FilesystemF2FSBase(Base):
    def generate_data_dict(self, data_str):
        data_str = data_str.replace(" = ", "=")
        data_str = data_str.replace(",", " ")
        return super(FilesystemF2FSBase, self).generate_data_dict(data_str)

    def finalize_object(self):
        self.data_frame.rename(columns={'ino':'inode'}, inplace=True)

class FilesystemF2FSWriteBegin(FilesystemF2FSBase):
    """Corresponds to Linux kernel trace event f2fs_write_begin"""

    unique_word = "f2fs_write_begin:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemF2FSWriteBegin)

class FilesystemF2FSWriteEnd(FilesystemF2FSBase):
    """Corresponds to Linux kernel trace event f2fs_write_end"""

    unique_word = "f2fs_write_end:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemF2FSWriteEnd)

class FilesystemF2FSSyncFileEnter(FilesystemF2FSBase):
    """Corresponds to Linux kernel trace event f2fs_sync_file_enter"""

    unique_word = "f2fs_sync_file_enter:"
    """The unique word that will be matched in a trace line"""

register_ftrace_parser(FilesystemF2FSSyncFileEnter)

class FilesystemF2FSSyncFileExit(FilesystemF2FSBase):
    """Corresponds to Linux kernel trace event f2fs_sync_file_exit"""

    unique_word = "f2fs_sync_file_exit:"
    """The unique word that will be matched in a trace line"""

    def generate_data_dict(self, data_str):
        data_str = data_str.replace("checkpoint is ", "checkpoint = ")
        return super(FilesystemF2FSSyncFileExit, self).generate_data_dict(data_str)

register_ftrace_parser(FilesystemF2FSSyncFileExit)
