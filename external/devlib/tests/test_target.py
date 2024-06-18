#
#    Copyright 2024 ARM Limited
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
Module for testing targets.

Sample run with log level is set to DEBUG (see
https://docs.pytest.org/en/7.1.x/how-to/logging.html#live-logs for logging details):

$ python -m pytest --log-cli-level DEBUG test_target.py
"""

import logging
import os
import pytest

from devlib import AndroidTarget, ChromeOsTarget, LinuxTarget, LocalLinuxTarget
from devlib._target_runner import NOPTargetRunner, QEMUTargetRunner
from devlib.utils.android import AdbConnection
from devlib.utils.misc import load_struct_from_yaml


logger = logging.getLogger('test_target')


def get_class_object(name):
    """
    Get associated class object from string formatted class name

    :param name: Class name
    :type name: str
    :return: Class object
    :rtype: object or None
    """
    if globals().get(name) is None:
        return None

    return globals()[name] if issubclass(globals()[name], object) else None


@pytest.fixture(scope="module")
# pylint: disable=too-many-branches
def build_target_runners():
    """Read targets from a YAML formatted config file and create runners for them"""

    logger.info("Initializing resources...")

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_config.yml')

    test_config = load_struct_from_yaml(config_file)
    if test_config is None:
        raise ValueError(f'{config_file} looks empty!')

    target_configs = test_config.get('target-configs')
    if target_configs is None:
        raise ValueError('No targets found!')

    target_runners = []

    for entry in target_configs.values():
        key, target_info = entry.popitem()

        target_class = get_class_object(key)
        if target_class is AndroidTarget:
            logger.info('> Android target: %s', repr(target_info))
            a_target = AndroidTarget(
                connect=False,
                connection_settings=target_info['connection_settings'],
                conn_cls=lambda **kwargs: AdbConnection(adb_as_root=True, **kwargs),
            )
            a_target.connect(timeout=target_info.get('timeout', 60))
            target_runners.append(NOPTargetRunner(a_target))

        elif target_class is ChromeOsTarget:
            logger.info('> ChromeOS target: %s', repr(target_info))
            c_target = ChromeOsTarget(
                connection_settings=target_info['connection_settings'],
                working_directory='/tmp/devlib-target',
            )
            target_runners.append(NOPTargetRunner(c_target))

        elif target_class is LinuxTarget:
            logger.info('> Linux target: %s', repr(target_info))
            l_target = LinuxTarget(connection_settings=target_info['connection_settings'])
            target_runners.append(NOPTargetRunner(l_target))

        elif target_class is LocalLinuxTarget:
            logger.info('> LocalLinux target: %s', repr(target_info))
            ll_target = LocalLinuxTarget(connection_settings=target_info['connection_settings'])
            target_runners.append(NOPTargetRunner(ll_target))

        elif target_class is QEMUTargetRunner:
            logger.info('> QEMU target runner: %s', repr(target_info))

            qemu_runner = QEMUTargetRunner(
                qemu_settings=target_info.get('qemu_settings'),
                connection_settings=target_info.get('connection_settings'),
            )

            if target_info.get('ChromeOsTarget') is not None:
                # Leave termination of QEMU runner to ChromeOS target.
                target_runners.append(NOPTargetRunner(qemu_runner.target))

                logger.info('>> ChromeOS target: %s', repr(target_info["ChromeOsTarget"]))
                qemu_runner.target = ChromeOsTarget(
                    connection_settings={
                        **target_info['ChromeOsTarget']['connection_settings'],
                        **qemu_runner.target.connection_settings,
                    },
                    working_directory='/tmp/devlib-target',
                )

            target_runners.append(qemu_runner)

        else:
            raise ValueError(f'Unknown target type {key}!')

    yield target_runners

    logger.info("Destroying resources...")

    for target_runner in target_runners:
        target = target_runner.target

        # TODO: Revisit per https://github.com/ARM-software/devlib/issues/680.
        logger.debug('Removing %s...', target.working_directory)
        target.remove(target.working_directory)

        target_runner.terminate()


# pylint: disable=redefined-outer-name
def test_read_multiline_values(build_target_runners):
    """
    Test Target.read_tree_values_flat()

    Runs tests around ``Target.read_tree_values_flat()`` for ``TargetRunner`` objects.
    """

    logger.info('Running test_read_multiline_values test...')

    data = {
        'test1': '1',
        'test2': '2\n\n',
        'test3': '3\n\n4\n\n',
    }

    target_runners = build_target_runners
    for target_runner in target_runners:
        target = target_runner.target

        logger.info('target=%s os=%s hostname=%s',
                    target.__class__.__name__, target.os, target.hostname)

        with target.make_temp() as tempdir:
            logger.debug('Created %s.', tempdir)

            for key, value in data.items():
                path = os.path.join(tempdir, key)
                logger.debug('Writing %s to %s...', repr(value), path)
                target.write_value(path, value, verify=False,
                                   as_root=target.conn.connected_as_root)

            logger.debug('Reading values from target...')
            raw_result = target.read_tree_values_flat(tempdir)
            result = {os.path.basename(k): v for k, v in raw_result.items()}

        assert {k: v.strip() for k, v in data.items()} == result
