# pylint: disable=attribute-defined-outside-init
import os
import re
import csv
import math
from collections import defaultdict

from wlauto import Workload, Parameter
from wlauto.exceptions import WorkloadError
from wlauto.utils.misc import check_output, get_null, get_meansd
from wlauto.utils.types import numeric, identifier


RESULT_REGEX = re.compile(r'RESULT (\w+): ([^=]+)\s*=\s*\[([^\]]+)\]\s*(\S+)')


class Telemetry(Workload):

    name = 'telemetry'
    description = """
    Executes Google's Telemetery benchmarking framework (must be installed).

    Url: https://www.chromium.org/developers/telemetry

    From the web site:

    Telemetry is Chrome's performance testing framework. It allows you to
    perform arbitrary actions on a set of web pages and report metrics about
    it. The framework abstracts:

      - Launching a browser with arbitrary flags on any platform.
      - Opening a tab and navigating to the page under test.
      - Fetching data via the Inspector timeline and traces.
      - Using Web Page Replay to cache real-world websites so they don't
        change when used in benchmarks.

    Design Principles

      - Write one performance test that runs on all platforms - Windows, Mac,
        Linux, Chrome OS, and Android for both Chrome and ContentShell.
      - Runs on browser binaries, without a full Chromium checkout, and without
        having to build the browser yourself.
      - Use WebPageReplay to get repeatable test results.
      - Clean architecture for writing benchmarks that keeps measurements and
        use cases separate.
      - Run on non-Chrome browsers for comparative studies.

    This instrument runs  telemetry via its ``run_benchmarks`` script (which
    must be in PATH or specified using ``run_benchmarks_path`` parameter) and
    parses metrics from the resulting output.

    **device setup**

    The device setup will depend on whether you're running a test image (in
    which case little or no setup should be necessary)


    """

    parameters = [
        Parameter('run_benchmark_path', default='run_benchmark',
                  description="""
                  This is the path to run_benchmark script which runs a
                  Telemetry benchmark. If not specified, the assumption will be
                  that it is in path (i.e. with be invoked as ``run_benchmark``).
                  """),
        Parameter('test', default='page_cycler.top_10_mobile',
                  description="""
                  Specifies with of the the telemetry tests is to be run.
                  """),
        Parameter('run_benchmark_params', default='',
                  description="""
                  Additional paramters to be passed to ``run_benchmarks``.
                  """),
        Parameter('run_timeout', kind=int, default=900,
                  description="""
                  Timeout for execution of the test.
                  """),
    ]

    summary_metrics = ['cold_times',
                       'commit_charge',
                       'cpu_utilization',
                       'processes',
                       'resident_set_size_peak_size_browser',
                       'resident_set_size_peak_size_gpu',
                       'vm_final_size_browser',
                       'vm_final_size_gpu',
                       'vm_final_size_renderer',
                       'vm_final_size_total',
                       'vm_peak_size_browser',
                       'vm_peak_size_gpu',
                       'vm_private_dirty_final_browser',
                       'vm_private_dirty_final_gpu',
                       'vm_private_dirty_final_renderer',
                       'vm_private_dirty_final_total',
                       'vm_resident_set_size_final_size_browser',
                       'vm_resident_set_size_final_size_gpu',
                       'vm_resident_set_size_final_size_renderer',
                       'vm_resident_set_size_final_size_total',
                       'warm_times']

    def validate(self):
        ret = os.system('{} > {} 2>&1'.format(self.run_benchmark_path, get_null()))
        if ret == 0xff00:  # is it supposed to be 0xff?
            pass  # telemetry found and appears to be installed properly.
        elif ret == 127:
            raise WorkloadError('run_benchmarks not found (did you specify correct run_benchmarks_path?)')
        else:
            raise WorkloadError('Unexected error from run_benchmarks: {}'.format(ret))

    def setup(self, context):
        self.raw_output = None
        self.command = self.build_command()

    def run(self, context):
        self.logger.debug(self.command)
        self.raw_output, _ = check_output(self.command, shell=True, timeout=self.run_timeout)

    def update_result(self, context):
        if not self.raw_output:
            self.logger.warning('Did not get run_benchmark output.')
            return
        raw_outfile = os.path.join(context.output_directory, 'telemetry_raw.out')
        with open(raw_outfile, 'w') as wfh:
            wfh.write(self.raw_output)
        context.add_artifact('telemetry-raw', raw_outfile, kind='raw')

        results = parse_telemetry_results(raw_outfile)
        csv_outfile = os.path.join(context.output_directory, 'telemetry.csv')
        averages = defaultdict(list)
        with open(csv_outfile, 'wb') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['kind', 'url', 'iteration', 'value', 'units'])
            for result in results:
                name_template = identifier('{}_{}_{{}}'.format(result.url, result.kind))
                averages[result.kind].append(result.average)
                context.result.add_metric(name_template.format('avg'), result.average,
                                          result.units, lower_is_better=True)
                context.result.add_metric(name_template.format('sd'), result.std,
                                          result.units, lower_is_better=True)
                writer.writerows(result.rows)
            context.add_artifact('telemetry', csv_outfile, kind='data')

        for kind, values in averages.iteritems():
            context.result.add_metric(kind, special_average(values), lower_is_better=True)

    def teardown(self, context):
        pass

    def build_command(self):
        if self.device.platform == 'chromeos':
            device_opts = '--remote={} --browser=cros-chrome'.format(self.device.host)
        else:
            raise WorkloadError('Currently, telemetry workload supports only ChromeOS devices.')
        return '{} {} {} {}'.format(self.run_benchmark_path,
                                    self.test,
                                    device_opts,
                                    self.run_benchmark_params)


class TelemetryResult(object):

    @property
    def average(self):
        return get_meansd(self.values)[0]

    @property
    def std(self):
        return get_meansd(self.values)[1]

    @property
    def rows(self):
        for i, v in enumerate(self.values):
            yield [self.kind, self.url, i, v, self.units]

    def __init__(self, kind=None, url=None, values=None, units=None):
        self.kind = kind
        self.url = url
        self.values = values or []
        self.units = units

    def __str__(self):
        return 'TR({kind},{url},{values},{units})'.format(**self.__dict__)

    __repr__ = __str__


def parse_telemetry_results(filepath):
    results = []
    with open(filepath) as fh:
        for line in fh:
            match = RESULT_REGEX.search(line)
            if match:
                result = TelemetryResult()
                result.kind = match.group(1)
                result.url = match.group(2)
                result.values = map(numeric, match.group(3).split(','))
                result.units = match.group(4)
                results.append(result)
    return results


def special_average(values):
    """Overall score calculation. Tries to accound for large differences
    between different pages."""
    negs = [v < 0 for v in values]
    abs_logs = [(av and math.log(av, 10) or av)
                for av in map(abs, values)]
    signed_logs = []
    for lv, n in zip(abs_logs, negs):
        if n:
            signed_logs.append(-lv)
        else:
            signed_logs.append(lv)
    return get_meansd(signed_logs)[0]


if __name__ == '__main__':
    import sys
    from pprint import pprint
    path = sys.argv[1]
    pprint(parse_telemetry_results(path))

