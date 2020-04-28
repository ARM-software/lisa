#! /usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
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

import argparse
import collections
import os
import subprocess
import sys
import tarfile
import multiprocessing.pool
import glob
import pickle
import shutil
import contextlib
from pathlib import Path
import traceback

from exekall.utils import DB_FILENAME

def print_exception(e):
    traceback.print_exception(type(e), e, e.__traceback__)

def stringify(seq):
    return [str(x) for x in seq]

def cleanup(path):
    """
    Remove a folder of file.
    """
    _path = str(path)
    with contextlib.suppress(FileNotFoundError):
        try:
            shutil.rmtree(_path)
        except NotADirectoryError:
            os.unlink(_path)
    return path


def download_artifacts(bisector_report, log_folder, exekall_db, extra_opts):
    subprocess.call([
        'bisector', 'report', bisector_report,
        '-oexport-logs={}'.format(log_folder),
        '-oexport-db={}'.format(exekall_db),
        '-odownload',
        *stringify(extra_opts),
    ])

def extract_tar(tar, dest):
    """
    Tar extraction worker.

    Only extracts files not already extracted.
    """

    with tarfile.open(str(tar)) as f:
        member_names = f.getnames()
        root = Path(os.path.commonpath(member_names))
        # Only extract once if the file has already been downloaded and
        # extracted before to save IO bandwith
        if not all((dest/name).exists() for name in member_names):
            f.extractall(str(dest))

    return dest/root

def exekall_run(artifact_dir, db_path, test_patterns, test_src_list, log_level, load_type, replay_uuid):
    """
    exekall run worker
    """
    cmd = [
        'exekall', 'run', *stringify(test_src_list),
	    '--load-db', str(db_path), '--artifact-dir', str(artifact_dir),
    ]
    if replay_uuid:
    	cmd.extend(['--replay', replay_uuid])
    else:
        cmd.extend([
            '--load-type', load_type,
    	    *('--select={}'.format(x) for x in test_patterns)
            ])

    if log_level:
        cmd += ['--log-level', log_level]

    # Capture output so it is not interleaved since multiple workers are
    # running at the same time.
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    # Exekall returns non-zero when a test failure occur, which is expected
    except subprocess.CalledProcessError as e:
        out = e.output

    return artifact_dir, out.decode('utf-8')

def exekall_merge(merged_artifact, db_path_list):
    subprocess.check_call([
        'exekall', 'merge',
        '-o', str(merged_artifact),
        *(str(path.parent) for path in db_path_list),
    ])

    return merged_artifact

def exekall_compare(ref_db_path, new_db_path):
    out = subprocess.check_output([
        'exekall', 'compare',
        str(ref_db_path), str(new_db_path),
        '--non-significant',
        '--remove-tag', 'board',
    ])
    return out.decode('utf-8')

def main(argv):
    parser = argparse.ArgumentParser(description="""
Re-execute a LISA test on a artifacts referenced by a bisector report, and
show the summary of changes in the results.

All `bisector` options need to come last on the command line.

EXAMPLE:
    Re-execute all the "test_task_placement" tests on the 10 first iterations
    of a hikey960 integration test.
    $ check-test-fix.py -s '*:test_task_placement' hikey960.report.yml.gz --cache --only behaviour -oiterations=1-10
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    load_group = parser.add_mutually_exclusive_group(required=True)
    load_group.add_argument('-s', '--select',
        action='append',
        metavar='EXEKALL_SELECT_PATTERN',
        help='Exekall run --select pattern to re-execute',
    )

    load_group.add_argument('--replay',
        metavar='EXEKALL_RESULT_UUID',
        help='See exekall run --replay and bisector report -oresult-uuid',
    )

    parser.add_argument('-t', '--python-src',
        metavar='EXEKALL_PYTHON_SOURCES',
        action='append',
        default=[os.path.expandvars('$LISA_HOME/lisa/tests')],
        help='Exekall run python modules',
    )

    parser.add_argument('--log-level',
        action='store_true',
        help='See exekall run --log-level',
    )

    parser.add_argument('--load-type',
        default='*TestBundle',
        help='See `exekall run --load-type`. It is ignored if --replay is used.',
    )

    parser.add_argument('bisector_report',
        metavar='BISECTOR_REPORT',
        help='Bisector report to use',
    )

    parser.add_argument('bisector_options',
        metavar='BISECTOR_EXTRA_OPTIONS',
        nargs=argparse.REMAINDER,
        default=[],
        help='Extra bisector options',
    )

    parser.add_argument('-j',
        metavar='NR_CPUS',
        type=int,
        default=os.cpu_count() + 2,
        help="""Number of processes to use for exekall run and tar extraction.
        Defaults to number of CPUs + 2""",
    )

    parser.add_argument('-w', '--work-area',
        type=Path,
        help="Work area to use when processing the artifacts",
    )

    args = parser.parse_args(argv)

    nr_cpus = args.j
    nr_cpus = nr_cpus if nr_cpus > 0 else 1
    bisector_extra_opts = args.bisector_options
    result_uuid = args.replay
    if result_uuid:
        bisector_extra_opts.append('-oresult-uuid={}'.format(result_uuid))
    bisector_report = args.bisector_report
    exekall_test_patterns = args.select
    if exekall_test_patterns:
        bisector_extra_opts.append('-otestcase={}'.format(
            ','.join(exekall_test_patterns)
        ))

    exekall_python_srcs = args.python_src
    exekall_log_level = args.log_level
    exekall_load_type = args.load_type
    exekall_replay_uuid = result_uuid

    work_area = args.work_area or Path('{}.check-test-fix'.format(Path(bisector_report).name))
    work_area.mkdir(exist_ok=True)

    # Read-back the options used the previous time. If they are the same,
    # no need to re-download an extract everything.
    extraction_state_path = str(work_area/'extraction_state')
    try:
        with open(extraction_state_path, 'rb') as f:
            extraction_state = pickle.load(f)
    except FileNotFoundError:
        need_download = True
    # If bisector options are different, we need to download and extract again
    else:
        need_download = (extraction_state['bisector_opts'] != bisector_extra_opts)

    if need_download:
        log_folder = work_area/'bisector_logs'
        ref_db_path = work_area/'reference_VALUE_DB.pickle.xz'

        # Make sure we don't have any archives from previous download, to avoid
        # selecting more archives than we want when listing the content of the
        # folder.
        cleanup(log_folder)

        download_artifacts(bisector_report, log_folder, ref_db_path,
            bisector_extra_opts)
    else:
        print('Reusing previously downloaded artifacts ...')
        ref_db_path = extraction_state['ref_db_path']
        extracted_artifacts = extraction_state['extracted_artifacts']
        log_folder = None


    # List of ValueDB paths that are freshly created by exekall run.
    # Since it is accessed by the pool's callback thread, use an atomic deque
    new_db_path_deque = collections.deque()

    # Use the context manager to make sure all worker subprocesses are
    # terminated
    with multiprocessing.pool.Pool(processes=nr_cpus) as exekall_pool:

        def exekall_run_callback(run_result):
            artifact_dir, out = run_result
            # Since all callbacks are called from the same thread,
            new_db_path_deque.append(artifact_dir/DB_FILENAME)

            # Show the log of the run in one go, to avoid interleaved output
            print('#'*80)
            print('New results for {}'.format(artifact_dir))
            print('\n\n')
            print(out)

        def schedule_exekall_run(artifact_dir):
            # schedule an `exekall run` job
            new_artifact_dir = cleanup(work_area/'new_exekall_artifacts'/artifact_dir.name)
            exekall_pool.apply_async(exekall_run,
                kwds=dict(
                    artifact_dir=new_artifact_dir,
                    db_path=artifact_dir/DB_FILENAME,
                    test_patterns=exekall_test_patterns,
                    test_src_list=exekall_python_srcs,
                    log_level=exekall_log_level,
                    load_type=exekall_load_type,
                    replay_uuid=exekall_replay_uuid,
                ),
                callback=exekall_run_callback,
                error_callback=print_exception,
            )

        # Uncompress all downloaded archives and kick off exekall run from the
        # callback
        if need_download:
            # Use a deque since it is appended from a different thread
            extracted_artifacts = collections.deque()

            def untar_callback(artifact_dir):
                extracted_artifacts.append(artifact_dir)
                return schedule_exekall_run(artifact_dir)

            # extract archives asynchronously
            with multiprocessing.pool.Pool(processes=nr_cpus) as untar_pool:
                for tar in glob.iglob(
                        str(log_folder/'**'/'lisa_artifacts.*'),
                        recursive=True
                    ):
                    untar_pool.apply_async(extract_tar,
                        kwds=dict(
                            tar=tar,
                            dest=work_area/'extracted_exekall_artifacts'
                        ),
                        callback=untar_callback,
                        error_callback=print_exception,
                    )

                untar_pool.close()
                untar_pool.join()
        # Kick of exekall run
        else:
            for artifact_dir in extracted_artifacts:
                schedule_exekall_run(artifact_dir)

        # We know that once all archives has been uncompressed, all exekall run
        # jobs have been submitted
        exekall_pool.close()
        exekall_pool.join()

        # Save what was extracted for future reference when we know all the
        # archives have been extracted successfully
        with open(extraction_state_path, 'wb') as f:
            pickle.dump({
                    'bisector_opts': bisector_extra_opts,
                    'ref_db_path': ref_db_path,
                    'extracted_artifacts': list(extracted_artifacts),
                },
                f
            )

    # Merge DB before comparison
    print('Merging the new artifacts for comparison ...')
    merged_artifact = cleanup(work_area/'merged_new_exekall_artifacts')
    merged_artifact_dir = exekall_merge(merged_artifact, new_db_path_deque)

    print('Comparing the results ...')
    out = exekall_compare(ref_db_path, merged_artifact_dir/DB_FILENAME)

    print('\n\nRegressions/improvements:')
    print(out)


if __name__ == '__main__':
    ret = main(sys.argv[1:])
    sys.exit(ret)

# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
