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

import os
import re
import logging


from builtins import zip  # pylint: disable=redefined-builtin
from future.moves.itertools import zip_longest

from wa.utils.misc import diff_tokens, write_table
from wa.utils.misc import ensure_file_directory_exists as _f

logger = logging.getLogger('diff')


def diff_interrupt_files(before, after, result):  # pylint: disable=R0914
    output_lines = []
    with open(before) as bfh:
        with open(after) as ofh:
            for bline, aline in zip(bfh, ofh):
                bchunks = bline.strip().split()
                while True:
                    achunks = aline.strip().split()
                    if achunks[0] == bchunks[0]:
                        diffchunks = ['']
                        diffchunks.append(achunks[0])
                        diffchunks.extend([diff_tokens(b, a) for b, a
                                           in zip(bchunks[1:], achunks[1:])])
                        output_lines.append(diffchunks)
                        break
                    else:  # new category appeared in the after file
                        diffchunks = ['>'] + achunks
                        output_lines.append(diffchunks)
                        try:
                            aline = next(ofh)
                        except StopIteration:
                            break

    # Offset heading columns by one to allow for row labels on subsequent
    # lines.
    output_lines[0].insert(0, '')

    # Any "columns" that do not have headings in the first row are not actually
    # columns -- they are a single column where space-spearated words got
    # split. Merge them back together to prevent them from being
    # column-aligned by write_table.
    table_rows = [output_lines[0]]
    num_cols = len(output_lines[0])
    for row in output_lines[1:]:
        table_row = row[:num_cols]
        table_row.append(' '.join(row[num_cols:]))
        table_rows.append(table_row)

    with open(result, 'w') as wfh:
        write_table(table_rows, wfh)


def diff_sysfs_dirs(before, after, result):  # pylint: disable=R0914
    before_files = []
    for root, _, files in os.walk(before):
        before_files.extend([os.path.join(root, f) for f in files])
    before_files = list(filter(os.path.isfile, before_files))
    files = [os.path.relpath(f, before) for f in before_files]
    after_files = [os.path.join(after, f) for f in files]
    diff_files = [os.path.join(result, f) for f in files]

    for bfile, afile, dfile in zip(before_files, after_files, diff_files):
        if not os.path.isfile(afile):
            logger.debug('sysfs_diff: {} does not exist or is not a file'.format(afile))
            continue

        with open(bfile) as bfh, open(afile) as afh:  # pylint: disable=C0321
            with open(_f(dfile), 'w') as dfh:
                for i, (bline, aline) in enumerate(zip_longest(bfh, afh), 1):
                    if aline is None:
                        logger.debug('Lines missing from {}'.format(afile))
                        break
                    bchunks = re.split(r'(\W+)', bline)
                    achunks = re.split(r'(\W+)', aline)
                    if len(bchunks) != len(achunks):
                        logger.debug('Token length mismatch in {} on line {}'.format(bfile, i))
                        dfh.write('xxx ' + bline)
                        continue
                    if ((len([c for c in bchunks if c.strip()]) == len([c for c in achunks if c.strip()]) == 2)
                            and (bchunks[0] == achunks[0])):
                        # if there are only two columns and the first column is the
                        # same, assume it's a "header" column and do not diff it.
                        dchunks = [bchunks[0]] + [diff_tokens(b, a) for b, a in zip(bchunks[1:], achunks[1:])]
                    else:
                        dchunks = [diff_tokens(b, a) for b, a in zip(bchunks, achunks)]
                    dfh.write(''.join(dchunks))
