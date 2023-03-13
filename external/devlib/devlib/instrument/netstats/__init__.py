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
import tempfile
from datetime import datetime
from collections import defaultdict

from future.moves.itertools import zip_longest

from devlib.instrument import Instrument, MeasurementsCsv, CONTINUOUS
from devlib.exception import TargetStableError, HostError
from devlib.utils.android import ApkInfo
from devlib.utils.csvutil import csvwriter


THIS_DIR = os.path.dirname(__file__)

NETSTAT_REGEX = re.compile(r'I/(?P<tag>netstats-\d+)\(\s*\d*\): (?P<ts>\d+) '
                           r'"(?P<package>[^"]+)" TX: (?P<tx>\S+) RX: (?P<rx>\S+)')


def extract_netstats(filepath, tag=None):
    netstats = []
    with open(filepath) as fh:
        for line in fh:
            match = NETSTAT_REGEX.search(line)
            if not match:
                continue
            if tag and match.group('tag') != tag:
                continue
            netstats.append((match.group('tag'),
                             match.group('ts'),
                             match.group('package'),
                             match.group('tx'),
                             match.group('rx')))
    return netstats


def netstats_to_measurements(netstats):
    measurements = defaultdict(list)
    for row in netstats:
        tag, ts, package, tx, rx = row  # pylint: disable=unused-variable
        measurements[package + '_tx'].append(tx)
        measurements[package + '_rx'].append(rx)
    return measurements


def write_measurements_csv(measurements, filepath):
    headers = sorted(measurements.keys())
    columns = [measurements[h] for h in headers]
    with csvwriter(filepath) as writer:
        writer.writerow(headers)
        writer.writerows(zip_longest(*columns))


class NetstatsInstrument(Instrument):

    mode = CONTINUOUS

    def __init__(self, target, apk=None, service='.TrafficMetricsService'):
        """
        Additional paramerter:

        :apk: Path to the APK file that contains ``com.arm.devlab.netstats``
              package. If not specified, it will be assumed that an APK with
              name "netstats.apk" is located in the same directory as the
              Python module for the instrument.
        :service: Name of the service to be launched. This service must be
                  present in the APK.

        """
        if target.os != 'android':
            raise TargetStableError('netstats instrument only supports Android targets')
        if apk is None:
            apk = os.path.join(THIS_DIR, 'netstats.apk')
        if not os.path.isfile(apk):
            raise HostError('APK for netstats instrument does not exist ({})'.format(apk))
        super(NetstatsInstrument, self).__init__(target)
        self.apk = apk
        self.package = ApkInfo(self.apk).package
        self.service = service
        self.tag = None
        self.command = None
        self.stop_command = 'am kill {}'.format(self.package)

        for package in self.target.list_packages():
            self.add_channel(package, 'tx')
            self.add_channel(package, 'rx')

    # pylint: disable=keyword-arg-before-vararg,arguments-differ
    def setup(self, force=False, *args, **kwargs):
        if self.target.package_is_installed(self.package):
            if force:
                self.logger.debug('Re-installing {} (forced)'.format(self.package))
                self.target.uninstall_package(self.package)
                self.target.install(self.apk)
            else:
                self.logger.debug('{} already present on target'.format(self.package))
        else:
            self.logger.debug('Deploying {} to target'.format(self.package))
            self.target.install(self.apk)

    def reset(self, sites=None, kinds=None, channels=None, period=None):  # pylint: disable=arguments-differ
        super(NetstatsInstrument, self).reset(sites, kinds, channels)
        period_arg, packages_arg = '', ''
        self.tag = 'netstats-{}'.format(datetime.now().strftime('%Y%m%d%H%M%s'))
        tag_arg = ' --es tag {}'.format(self.tag)
        if sites:
            packages_arg = ' --es packages {}'.format(','.join(sites))
        if period:
            period_arg = ' --ei period {}'.format(period)
        self.command = 'am startservice{}{}{} {}/{}'.format(tag_arg,
                                                            period_arg,
                                                            packages_arg,
                                                            self.package,
                                                            self.service)
        self.target.execute(self.stop_command)  # ensure the service is not running.

    def start(self):
        if self.command is None:
            raise RuntimeError('reset() must be called before start()')
        self.target.execute(self.command)

    def stop(self):
        self.target.execute(self.stop_command)

    def get_data(self, outfile):
        raw_log_file = tempfile.mktemp()
        self.target.dump_logcat(raw_log_file)
        data = extract_netstats(raw_log_file)
        measurements = netstats_to_measurements(data)
        write_measurements_csv(measurements, outfile)
        os.remove(raw_log_file)
        return MeasurementsCsv(outfile, self.active_channels)

    def teardown(self):
        self.target.uninstall_package(self.package)
