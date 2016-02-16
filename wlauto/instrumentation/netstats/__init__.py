import os
import re
import csv
import tempfile
import logging
from datetime import datetime
from collections import defaultdict
from itertools import izip_longest

from wlauto import Instrument, Parameter
from wlauto import ApkFile
from wlauto.exceptions import DeviceError, HostError
from wlauto.utils.android import ApkInfo
from wlauto.utils.types import list_of_strings


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
    with open(filepath, 'wb') as wfh:
        writer = csv.writer(wfh)
        writer.writerow(headers)
        writer.writerows(izip_longest(*columns))


class NetstatsCollector(object):

    def __init__(self, target, apk, service='.TrafficMetricsService'):
        """
        Additional paramerter:

        :apk: Path to the APK file that contains ``com.arm.devlab.netstats``
              package. If not specified, it will be assumed that an APK with
              name "netstats.apk" is located in the same directory as the
              Python module for the instrument.
        :service: Name of the service to be launched. This service must be
                  present in the APK.

        """
        self.target = target
        self.apk = apk
        self.logger = logging.getLogger('netstat')
        self.package = ApkInfo(self.apk).package
        self.service = service
        self.tag = None
        self.command = None
        self.stop_command = 'am kill {}'.format(self.package)

    def setup(self, force=False):
        if self.target.package_is_installed(self.package):
            if force:
                self.logger.debug('Re-installing {} (forced)'.format(self.package))
                self.target.uninstall(self.package)
                self.target.install(self.apk, timeout=300)
            else:
                self.logger.debug('{} already present on target'.format(self.package))
        else:
            self.logger.debug('Deploying {} to target'.format(self.package))
            self.target.install(self.apk)

    def reset(self, sites=None, period=None):
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

    def teardown(self):
        self.target.uninstall(self.package)


class NetstatsInstrument(Instrument):
    # pylint: disable=unused-argument

    name = 'netstats'
    description = """
    Measures transmit/receive network traffic on an Android divice on per-package
    basis.

    """

    parameters = [
        Parameter('packages', kind=list_of_strings,
                  description="""
                  List of Android packages who's traffic will be monitored. If
                  unspecified, all packages in the device will be monitorred.
                  """),
        Parameter('period', kind=int, default=5,
                  description="""
                  Polling period for instrumentation on the device. Traffic statistics
                  will be updated every ``period`` seconds.
                  """),
        Parameter('force_reinstall', kind=bool, default=False,
                  description="""
                  If ``True``, instrumentation APK will always be re-installed even if
                  it already installed on the device.
                  """),
        Parameter('uninstall_on_completion', kind=bool, default=False,
                  global_alias='cleanup',
                  description="""
                  If ``True``, instrumentation will be uninstalled upon run completion.
                  """),
    ]

    def initialize(self, context):
        if self.device.os != 'android':
            raise DeviceError('nestats instrument only supports on Android devices.')
        apk = context.resolver.get(ApkFile(self))
        self.collector = NetstatsCollector(self.device, apk)  # pylint: disable=attribute-defined-outside-init
        self.collector.setup(force=self.force_reinstall)

    def setup(self, context):
        self.collector.reset(sites=self.packages, period=self.period)

    def start(self, context):
        self.collector.start()

    def stop(self, context):
        self.collector.stop()

    def update_result(self, context):
        outfile = os.path.join(context.output_directory, 'netstats.csv')
        self.collector.get_data(outfile)
        context.add_artifact('netstats', outfile, kind='data')
        with open(outfile, 'rb') as fh:
            reader = csv.reader(fh)
            metrics = reader.next()
            data = [c for c in izip_longest(*list(reader))]
            for name, values in zip(metrics, data):
                value = sum(map(int, [v for v in values if v]))
                context.add_metric(name, value, units='bytes')

    def finalize(self, context):
        if self.uninstall_on_completion:
            self.collector.teardown()

