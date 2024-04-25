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

"""Module for testing targets."""

import os
from pprint import pp
import pytest

from devlib import AndroidTarget, ChromeOsTarget, LinuxTarget, LocalLinuxTarget, QEMUTargetRunner
from devlib.utils.android import AdbConnection
from devlib.utils.misc import load_struct_from_yaml


def build_targets():
    """Read targets from a YAML formatted config file"""

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'target_configs.yaml')

    target_configs = load_struct_from_yaml(config_file)
    if target_configs is None:
        raise ValueError(f'{config_file} looks empty!')

    targets = []

    if target_configs.get('AndroidTarget') is not None:
        print('> Android targets:')
        for entry in target_configs['AndroidTarget'].values():
            pp(entry)
            a_target = AndroidTarget(
                connect=False,
                connection_settings=entry['connection_settings'],
                conn_cls=lambda **kwargs: AdbConnection(adb_as_root=True, **kwargs),
            )
            a_target.connect(timeout=entry.get('timeout', 60))
            targets.append((a_target, None))

    if target_configs.get('LinuxTarget') is not None:
        print('> Linux targets:')
        for entry in target_configs['LinuxTarget'].values():
            pp(entry)
            l_target = LinuxTarget(connection_settings=entry['connection_settings'])
            targets.append((l_target, None))

    if target_configs.get('ChromeOsTarget') is not None:
        print('> ChromeOS targets:')
        for entry in target_configs['ChromeOsTarget'].values():
            pp(entry)
            c_target = ChromeOsTarget(
                connection_settings=entry['connection_settings'],
                working_directory='/tmp/devlib-target',
            )
            targets.append((c_target, None))

    if target_configs.get('LocalLinuxTarget') is not None:
        print('> LocalLinux targets:')
        for entry in target_configs['LocalLinuxTarget'].values():
            pp(entry)
            ll_target = LocalLinuxTarget(connection_settings=entry['connection_settings'])
            targets.append((ll_target, None))

    if target_configs.get('QEMUTargetRunner') is not None:
        print('> QEMU target runners:')
        for entry in target_configs['QEMUTargetRunner'].values():
            pp(entry)
            qemu_settings = entry.get('qemu_settings') and entry['qemu_settings']
            connection_settings = entry.get(
                'connection_settings') and entry['connection_settings']

            qemu_runner = QEMUTargetRunner(
                qemu_settings=qemu_settings,
                connection_settings=connection_settings,
            )

            if entry.get('ChromeOsTarget') is None:
                targets.append((qemu_runner.target, qemu_runner))
                continue

            # Leave termination of QEMU runner to ChromeOS target.
            targets.append((qemu_runner.target, None))

            print('> ChromeOS targets:')
            pp(entry['ChromeOsTarget'])
            c_target = ChromeOsTarget(
                connection_settings={
                    **entry['ChromeOsTarget']['connection_settings'],
                    **qemu_runner.target.connection_settings,
                },
                working_directory='/tmp/devlib-target',
            )
            targets.append((c_target, qemu_runner))

    return targets


@pytest.mark.parametrize("target, target_runner", build_targets())
def test_read_multiline_values(target, target_runner):
    """
    Test Target.read_tree_values_flat()

    :param target: Type of target per :class:`Target` based classes.
    :type target: Target

    :param target_runner: Target runner object to terminate target (if necessary).
    :type target: TargetRunner
    """

    data = {
        'test1': '1',
        'test2': '2\n\n',
        'test3': '3\n\n4\n\n',
    }

    print(f'target={target.__class__.__name__} os={target.os} hostname={target.hostname}')

    with target.make_temp() as tempdir:
        print(f'Created {tempdir}.')

        for key, value in data.items():
            path = os.path.join(tempdir, key)
            print(f'Writing {value!r} to {path}...')
            target.write_value(path, value, verify=False,
                               as_root=target.conn.connected_as_root)

        print('Reading values from target...')
        raw_result = target.read_tree_values_flat(tempdir)
        result = {os.path.basename(k): v for k, v in raw_result.items()}

    print(f'Removing {target.working_directory}...')
    target.remove(target.working_directory)

    if target_runner is not None:
        print('Terminating target runner...')
        target_runner.terminate()

    assert {k: v.strip() for k, v in data.items()} == result
