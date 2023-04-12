#    Copyright 2018 ARM Limited
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

'''
Due to the change in the nature of "binary mode" when opening files in
Python 3, the way files need to be opened for ``csv.reader`` and ``csv.writer``
is different from Python 2.

The functions in this module are intended to hide these differences allowing
the rest of the code to create csv readers/writers without worrying about which
Python version it is running under.

First up are ``csvwriter`` and ``csvreader`` context mangers that handle the
opening and closing of the underlying file. These are intended to replace the
most common usage pattern

.. code-block:: python

    with open(filepath, 'wb') as wfh:  # or open(filepath, 'w', newline='') in Python 3
        writer = csv.writer(wfh)
        writer.writerows(data)


with

.. code-block:: python

    with csvwriter(filepath) as writer:
        writer.writerows(data)


``csvreader`` works in an analogous way. ``csvreader`` and ``writer`` can take
additional arguments which will be passed directly to the
``csv.reader``/``csv.writer`` calls.

In some cases, it is desirable not to use a context manager (e.g. if the
reader/writer is intended to be returned from the function that creates it. For
such cases, alternative functions, ``create_reader`` and ``create_writer``,
exit. These return a two-tuple, with the created reader/writer as the first
element, and the corresponding ``FileObject`` as the second. It is the
responsibility of the calling code to ensure that the file is closed properly.

'''
import csv
import sys
from contextlib import contextmanager


@contextmanager
def csvwriter(filepath, *args, **kwargs):
    if sys.version_info[0] == 3:
        wfh = open(filepath, 'w', newline='')
    else:
        wfh = open(filepath, 'wb')

    try:
        yield csv.writer(wfh, *args, **kwargs)
    finally:
        wfh.close()


@contextmanager
def csvreader(filepath, *args, **kwargs):
    if sys.version_info[0] == 3:
        fh = open(filepath, 'r', newline='')
    else:
        fh = open(filepath, 'rb')

    try:
        yield csv.reader(fh, *args, **kwargs)
    finally:
        fh.close()


def create_writer(filepath, *args, **kwargs):
    if sys.version_info[0] == 3:
        wfh = open(filepath, 'w', newline='')
    else:
        wfh = open(filepath, 'wb')
    return csv.writer(wfh, *args, **kwargs), wfh


def create_reader(filepath, *args, **kwargs):
    if sys.version_info[0] == 3:
        fh = open(filepath, 'r', newline='')
    else:
        fh = open(filepath, 'rb')
    return csv.reader(fh, *args, **kwargs), fh
