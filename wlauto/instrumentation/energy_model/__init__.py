#    Copyright 2015 ARM Limited
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


#pylint: disable=attribute-defined-outside-init,access-member-before-definition,redefined-outer-name
from __future__ import division
import os
import math
import time
from tempfile import mktemp
from base64 import b64encode
from collections import Counter, namedtuple

try:
    import jinja2
    import pandas as pd
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
    import numpy as np
    low_filter = np.vectorize(lambda x: x > 0 and x or 0)  # pylint: disable=no-member
    import_error = None
except ImportError as e:
    import_error = e
    jinja2 = None
    pd = None
    plt = None
    np = None
    low_filter = None

from wlauto import Instrument, Parameter, File
from wlauto.exceptions import ConfigError, InstrumentError, DeviceError
from wlauto.instrumentation import instrument_is_installed
from wlauto.utils.types import caseless_string, list_or_caseless_string, list_of_ints
from wlauto.utils.misc import list_to_mask

FREQ_TABLE_FILE = 'frequency_power_perf_data.csv'
CPUS_TABLE_FILE = 'projected_cap_power.csv'
MEASURED_CPUS_TABLE_FILE = 'measured_cap_power.csv'
IDLE_TABLE_FILE = 'idle_power_perf_data.csv'
REPORT_TEMPLATE_FILE = 'report.template'
EM_TEMPLATE_FILE = 'em.template'

IdlePowerState = namedtuple('IdlePowerState', ['power'])
CapPowerState = namedtuple('CapPowerState', ['cap', 'power'])


class EnergyModel(object):

    def __init__(self):
        self.big_cluster_idle_states = []
        self.little_cluster_idle_states = []
        self.big_cluster_cap_states = []
        self.little_cluster_cap_states = []
        self.big_core_idle_states = []
        self.little_core_idle_states = []
        self.big_core_cap_states = []
        self.little_core_cap_states = []

    def add_cap_entry(self, cluster, perf, clust_pow, core_pow):
        if cluster == 'big':
            self.big_cluster_cap_states.append(CapPowerState(perf, clust_pow))
            self.big_core_cap_states.append(CapPowerState(perf, core_pow))
        elif cluster == 'little':
            self.little_cluster_cap_states.append(CapPowerState(perf, clust_pow))
            self.little_core_cap_states.append(CapPowerState(perf, core_pow))
        else:
            raise ValueError('Unexpected cluster: {}'.format(cluster))

    def add_cluster_idle(self, cluster, values):
        for value in values:
            if cluster == 'big':
                self.big_cluster_idle_states.append(IdlePowerState(value))
            elif cluster == 'little':
                self.little_cluster_idle_states.append(IdlePowerState(value))
            else:
                raise ValueError('Unexpected cluster: {}'.format(cluster))

    def add_core_idle(self, cluster, values):
        for value in values:
            if cluster == 'big':
                self.big_core_idle_states.append(IdlePowerState(value))
            elif cluster == 'little':
                self.little_core_idle_states.append(IdlePowerState(value))
            else:
                raise ValueError('Unexpected cluster: {}'.format(cluster))


class PowerPerformanceAnalysis(object):

    def __init__(self, data):
        self.summary = {}
        big_freqs = data[data.cluster == 'big'].frequency.unique()
        little_freqs = data[data.cluster == 'little'].frequency.unique()
        self.summary['frequency'] = max(set(big_freqs).intersection(set(little_freqs)))

        big_sc = data[(data.cluster == 'big') &
                      (data.frequency == self.summary['frequency']) &
                      (data.cpus == 1)]
        little_sc = data[(data.cluster == 'little') &
                         (data.frequency == self.summary['frequency']) &
                         (data.cpus == 1)]
        self.summary['performance_ratio'] = big_sc.performance.item() / little_sc.performance.item()
        self.summary['power_ratio'] = big_sc.power.item() / little_sc.power.item()
        self.summary['max_performance'] = data[data.cpus == 1].performance.max()
        self.summary['max_power'] = data[data.cpus == 1].power.max()


def build_energy_model(freq_power_table, cpus_power, idle_power, first_cluster_idle_state):
    # pylint: disable=too-many-locals
    em = EnergyModel()
    idle_power_sc = idle_power[idle_power.cpus == 1]
    perf_data = get_normalized_single_core_data(freq_power_table)

    for cluster in ['little', 'big']:
        cluster_cpus_power = cpus_power[cluster].dropna()
        cluster_power = cluster_cpus_power['cluster'].apply(int)
        core_power = (cluster_cpus_power['1'] - cluster_power).apply(int)
        performance = (perf_data[perf_data.cluster == cluster].performance_norm * 1024 / 100).apply(int)
        for perf, clust_pow, core_pow in zip(performance, cluster_power, core_power):
            em.add_cap_entry(cluster, perf, clust_pow, core_pow)

        all_idle_power = idle_power_sc[idle_power_sc.cluster == cluster].power.values
        # CORE idle states
        # We want the delta of each state w.r.t. the power
        # consumption of the shallowest one at this level (core_ref)
        idle_core_power = low_filter(all_idle_power[:first_cluster_idle_state] -
                                     all_idle_power[first_cluster_idle_state - 1])
        # CLUSTER idle states
        # We want the absolute value of each idle state
        idle_cluster_power = low_filter(all_idle_power[first_cluster_idle_state - 1:])
        em.add_cluster_idle(cluster, idle_cluster_power)
        em.add_core_idle(cluster, idle_core_power)

    return em


def generate_em_c_file(em, big_core, little_core, em_template_file, outfile):
    with open(em_template_file) as fh:
        em_template = jinja2.Template(fh.read())
    em_text = em_template.render(
        big_core=big_core,
        little_core=little_core,
        em=em,
    )
    with open(outfile, 'w') as wfh:
        wfh.write(em_text)
    return em_text


def generate_report(freq_power_table, measured_cpus_table, cpus_table, idle_power_table,  # pylint: disable=unused-argument
                    report_template_file, device_name, em_text, outfile):
    # pylint: disable=too-many-locals
    cap_power_analysis = PowerPerformanceAnalysis(freq_power_table)
    single_core_norm = get_normalized_single_core_data(freq_power_table)
    cap_power_plot = get_cap_power_plot(single_core_norm)
    idle_power_plot = get_idle_power_plot(idle_power_table)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(16, 8)
    for i, cluster in enumerate(reversed(cpus_table.columns.levels[0])):
        projected = cpus_table[cluster].dropna(subset=['1'])
        plot_cpus_table(projected, axes[i], cluster)
    cpus_plot_data = get_figure_data(fig)

    with open(report_template_file) as fh:
        report_template = jinja2.Template(fh.read())
    html = report_template.render(
        device_name=device_name,
        freq_power_table=freq_power_table.set_index(['cluster', 'cpus', 'frequency']).to_html(),
        cap_power_analysis=cap_power_analysis,
        cap_power_plot=get_figure_data(cap_power_plot),
        idle_power_table=idle_power_table.set_index(['cluster', 'cpus', 'state']).to_html(),
        idle_power_plot=get_figure_data(idle_power_plot),
        cpus_table=cpus_table.to_html(),
        cpus_plot=cpus_plot_data,
        em_text=em_text,
    )
    with open(outfile, 'w') as wfh:
        wfh.write(html)
    return html


def wa_result_to_power_perf_table(df, performance_metric, index):
    table = df.pivot_table(index=index + ['iteration'],
                           columns='metric', values='value').reset_index()
    result_mean = table.groupby(index).mean()
    result_std = table.groupby(index).std()
    result_std.columns = [c + ' std' for c in result_std.columns]
    result_count = table.groupby(index).count()
    result_count.columns = [c + ' count' for c in result_count.columns]
    count_sqrt = result_count.apply(lambda x: x.apply(math.sqrt))
    count_sqrt.columns = result_std.columns  # match column names for division
    result_error = 1.96 * result_std / count_sqrt  # 1.96 == 95% confidence interval
    result_error.columns = [c + ' error' for c in result_mean.columns]

    result = pd.concat([result_mean, result_std, result_count, result_error], axis=1)
    del result['iteration']
    del result['iteration std']
    del result['iteration count']
    del result['iteration error']

    updated_columns = []
    for column in result.columns:
        if column == performance_metric:
            updated_columns.append('performance')
        elif column == performance_metric + ' std':
            updated_columns.append('performance_std')
        elif column == performance_metric + ' error':
            updated_columns.append('performance_error')
        else:
            updated_columns.append(column.replace(' ', '_'))
    result.columns = updated_columns
    result = result[sorted(result.columns)]
    result.reset_index(inplace=True)

    return result


def get_figure_data(fig, fmt='png'):
    tmp = mktemp()
    fig.savefig(tmp, format=fmt, bbox_inches='tight')
    with open(tmp, 'rb') as fh:
        image_data = b64encode(fh.read())
    os.remove(tmp)
    return image_data


def get_normalized_single_core_data(data):
    finite_power = np.isfinite(data.power)  # pylint: disable=no-member
    finite_perf = np.isfinite(data.performance)  # pylint: disable=no-member
    data_single_core = data[(data.cpus == 1) & finite_perf & finite_power].copy()
    data_single_core['performance_norm'] = (data_single_core.performance /
                                            data_single_core.performance.max() * 100).apply(int)
    data_single_core['power_norm'] = (data_single_core.power /
                                      data_single_core.power.max() * 100).apply(int)
    return data_single_core


def get_cap_power_plot(data_single_core):
    big_single_core = data_single_core[(data_single_core.cluster == 'big') &
                                       (data_single_core.cpus == 1)]
    little_single_core = data_single_core[(data_single_core.cluster == 'little') &
                                          (data_single_core.cpus == 1)]

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    axes.plot(big_single_core.performance_norm,
              big_single_core.power_norm,
              marker='o')
    axes.plot(little_single_core.performance_norm,
              little_single_core.power_norm,
              marker='o')
    axes.set_xlim(0, 105)
    axes.set_ylim(0, 105)
    axes.set_xlabel('Performance (Normalized)')
    axes.set_ylabel('Power (Normalized)')
    axes.grid()
    axes.legend(['big cluster', 'little cluster'], loc=0)
    return fig


def get_idle_power_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for cluster, ax in zip(['little', 'big'], axes):
        data = df[df.cluster == cluster].pivot_table(index=['state'], columns='cpus', values='power')
        err = df[df.cluster == cluster].pivot_table(index=['state'], columns='cpus', values='power_error')
        data.plot(kind='bar', ax=ax, rot=30, yerr=err)
        ax.set_title('{} cluster'.format(cluster))
        ax.set_xlim(-1, len(data.columns) - 0.5)
        ax.set_ylabel('Power (mW)')
    return fig


def fit_polynomial(s, n):
    # pylint: disable=no-member
    coeffs = np.polyfit(s.index, s.values, n)
    poly = np.poly1d(coeffs)
    return poly(s.index)


def get_cpus_power_table(data, index, opps, leak_factors):  # pylint: disable=too-many-locals
    # pylint: disable=no-member
    power_table = data[[index, 'cluster', 'cpus', 'power']].pivot_table(index=index,
                                                                        columns=['cluster', 'cpus'],
                                                                        values='power')
    bs_power_table = pd.DataFrame(index=power_table.index, columns=power_table.columns)
    for cluster in power_table.columns.levels[0]:
        power_table[cluster, 0] = (power_table[cluster, 1] -
                                   (power_table[cluster, 2] -
                                    power_table[cluster, 1]))
        bs_power_table.loc[power_table[cluster, 1].notnull(), (cluster, 1)] = fit_polynomial(power_table[cluster, 1].dropna(), 2)
        bs_power_table.loc[power_table[cluster, 2].notnull(), (cluster, 2)] = fit_polynomial(power_table[cluster, 2].dropna(), 2)

        if opps[cluster] is None:
            bs_power_table.loc[bs_power_table[cluster, 1].notnull(), (cluster, 0)] = \
                (2 * power_table[cluster, 1] - power_table[cluster, 2]).values
        else:
            voltages = opps[cluster].set_index('frequency').sort_index()
            leakage = leak_factors[cluster] * 2 * voltages['voltage']**3 / 0.9**3
            leakage_delta = leakage - leakage[leakage.index[0]]
            bs_power_table.loc[:, (cluster, 0)] = \
                (2 * bs_power_table[cluster, 1] + leakage_delta - bs_power_table[cluster, 2])

    # re-order columns and rename colum '0' to  'cluster'
    power_table = power_table[sorted(power_table.columns,
                                     cmp=lambda x, y: cmp(y[0], x[0]) or cmp(x[1], y[1]))]
    bs_power_table = bs_power_table[sorted(bs_power_table.columns,
                                           cmp=lambda x, y: cmp(y[0], x[0]) or cmp(x[1], y[1]))]
    old_levels = power_table.columns.levels
    power_table.columns.set_levels([old_levels[0], list(map(str, old_levels[1])[:-1]) + ['cluster']],
                                   inplace=True)
    bs_power_table.columns.set_levels([old_levels[0], list(map(str, old_levels[1])[:-1]) + ['cluster']],
                                      inplace=True)
    return power_table, bs_power_table


def plot_cpus_table(projected, ax, cluster):
    projected.T.plot(ax=ax, marker='o')
    ax.set_title('{} cluster'.format(cluster))
    ax.set_xticklabels(projected.columns)
    ax.set_xticks(range(0, 5))
    ax.set_xlim(-0.5, len(projected.columns) - 0.5)
    ax.set_ylabel('Power (mW)')
    ax.grid(True)


def opp_table(d):
    if d is None:
        return None
    return pd.DataFrame(d.items(), columns=['frequency', 'voltage'])


class EnergyModelInstrument(Instrument):

    name = 'energy_model'
    desicription = """
    Generates a power mode for the device based on specified workload.

    This insturment will execute the workload specified by the agenda (currently, only ``sysbench`` is
    supported) and will use the resulting performance and power measurments to generate a power mode for
    the device.

    This instrument requires certain features to be present in the kernel:

    1. cgroups and cpusets must be enabled.
    2. cpufreq and userspace governor must be enabled.
    3. cpuidle must be enabled.

    """

    parameters = [
        Parameter('device_name', kind=caseless_string,
                  description="""The name of the device to be used in  generating the model. If not specified,
                                 ``device.name`` will be used. """),
        Parameter('big_core', kind=caseless_string,
                  description="""The name of the "big" core in the big.LITTLE system; must match
                                 one of the values in ``device.core_names``. """),
        Parameter('performance_metric', kind=caseless_string, mandatory=True,
                  description="""Metric to be used as the performance indicator."""),
        Parameter('power_metric', kind=list_or_caseless_string,
                  description="""Metric to be used as the power indicator. The value may contain a
                                 ``{core}`` format specifier that will be replaced with names of big
                                 and little cores to drive the name of the metric for that cluster.
                                 Ether this or ``energy_metric`` must be specified but not both."""),
        Parameter('energy_metric', kind=list_or_caseless_string,
                  description="""Metric to be used as the energy indicator. The value may contain a
                                 ``{core}`` format specifier that will be replaced with names of big
                                 and little cores to drive the name of the metric for that cluster.
                                 this metric will be used to derive power by deviding through by
                                 execution time. Either this or ``power_metric`` must be specified, but
                                 not both."""),
        Parameter('power_scaling_factor', kind=float, default=1.0,
                  description="""Power model specfies power in milliWatts. This is a scaling factor that
                                 power_metric values will be multiplied by to get milliWatts."""),
        Parameter('big_frequencies', kind=list_of_ints,
                  description="""List of frequencies to be used for big cores. These frequencies must
                                 be supported by the cores. If this is not specified, all available
                                 frequencies for the core (as read from cpufreq) will be used."""),
        Parameter('little_frequencies', kind=list_of_ints,
                  description="""List of frequencies to be used for little cores. These frequencies must
                                 be supported by the cores. If this is not specified, all available
                                 frequencies for the core (as read from cpufreq) will be used."""),
        Parameter('idle_workload', kind=str, default='idle',
                  description="Workload to be used while measuring idle power."),
        Parameter('idle_workload_params', kind=dict, default={},
                  description="Parameter to pass to the idle workload."),
        Parameter('first_cluster_idle_state', kind=int, default=-1,
                  description='''The index of the first cluster idle state on the device. Previous states
                                 are assumed to be core idles. The default is ``-1``, i.e. only the last
                                 idle state is assumed to affect the entire cluster.'''),
        Parameter('no_hotplug', kind=bool, default=False,
                  description='''This options allows running the instrument without hotpluging cores on and off.
                                 Disabling hotplugging will most likely produce a less accurate power model.'''),
        Parameter('num_of_freqs_to_thermal_adjust', kind=int, default=0,
                  description="""The number of frequencies begining from the highest, to be adjusted for
                                 the thermal effect."""),
        Parameter('big_opps', kind=opp_table,
                  description="""OPP table mapping frequency to voltage (kHz --> mV) for the big cluster."""),
        Parameter('little_opps', kind=opp_table,
                  description="""OPP table mapping frequency to voltage (kHz --> mV) for the little cluster."""),
        Parameter('big_leakage', kind=int, default=120,
                  description="""
                  Leakage factor for the big cluster (this is specific to a particular core implementation).
                  """),
        Parameter('little_leakage', kind=int, default=60,
                  description="""
                  Leakage factor for the little cluster (this is specific to a particular core implementation).
                  """),
    ]

    def validate(self):
        if import_error:
            message = 'energy_model instrument requires pandas, jinja2 and matplotlib Python packages to be installed; got: "{}"'
            raise InstrumentError(message.format(import_error.message))
        for capability in ['cgroups', 'cpuidle']:
            if not self.device.has(capability):
                message = 'The Device does not appear to support {}; does it have the right module installed?'
                raise ConfigError(message.format(capability))
        device_cores = set(self.device.core_names)
        if (self.power_metric and self.energy_metric) or not (self.power_metric or self.energy_metric):
            raise ConfigError('Either power_metric or energy_metric must be specified (but not both).')
        if not device_cores:
            raise ConfigError('The Device does not appear to have core_names configured.')
        elif len(device_cores) != 2:
            raise ConfigError('The Device does not appear to be a big.LITTLE device.')
        if self.big_core and self.big_core not in self.device.core_names:
            raise ConfigError('Specified big_core "{}" is in divice {}'.format(self.big_core, self.device.name))
        if not self.big_core:
            self.big_core = self.device.core_names[-1]  # the last core is usually "big" in existing big.LITTLE devices
        if not self.device_name:
            self.device_name = self.device.name
        if self.num_of_freqs_to_thermal_adjust and not instrument_is_installed('daq'):
            self.logger.warn('Adjustment for thermal effect requires daq instrument. Disabling adjustment')
            self.num_of_freqs_to_thermal_adjust = 0

    def initialize(self, context):
        self.number_of_cpus = {}
        self.report_template_file = context.resolver.get(File(self, REPORT_TEMPLATE_FILE))
        self.em_template_file = context.resolver.get(File(self, EM_TEMPLATE_FILE))
        self.little_core = (set(self.device.core_names) - set([self.big_core])).pop()
        self.perform_runtime_validation()
        self.enable_all_cores()
        self.configure_clusters()
        self.discover_idle_states()
        self.disable_thermal_management()
        self.initialize_job_queue(context)
        self.initialize_result_tracking()

    def setup(self, context):
        if not context.spec.label.startswith('idle_'):
            return
        for idle_state in self.get_device_idle_states(self.measured_cluster):
            if idle_state.index > context.spec.idle_state_index:
                idle_state.disable = 1
            else:
                idle_state.disable = 0

    def fast_start(self, context):  # pylint: disable=unused-argument
        self.start_time = time.time()

    def fast_stop(self, context):  # pylint: disable=unused-argument
        self.run_time = time.time() - self.start_time

    def on_iteration_start(self, context):
        self.setup_measurement(context.spec.cluster)

    def thermal_correction(self, context):
        if not self.num_of_freqs_to_thermal_adjust or self.num_of_freqs_to_thermal_adjust > len(self.big_frequencies):
            return 0
        freqs = self.big_frequencies[-self.num_of_freqs_to_thermal_adjust:]
        spec = context.result.spec
        if spec.frequency not in freqs:
            return 0
        data_path = os.path.join(context.output_directory, 'daq', '{}.csv'.format(self.big_core))
        data = pd.read_csv(data_path)['power']
        return _adjust_for_thermal(data, filt_method=lambda x: pd.rolling_median(x, 1000), thresh=0.9, window=5000)

    # slow to make sure power results have been generated
    def slow_update_result(self, context):  # pylint: disable=too-many-branches
        spec = context.result.spec
        cluster = spec.cluster
        is_freq_iteration = spec.label.startswith('freq_')
        perf_metric = 0
        power_metric = 0
        thermal_adjusted_power = 0
        if is_freq_iteration and cluster == 'big':
            thermal_adjusted_power = self.thermal_correction(context)
        for metric in context.result.metrics:
            if metric.name == self.performance_metric:
                perf_metric = metric.value
            elif thermal_adjusted_power and metric.name in self.big_power_metrics:
                power_metric += thermal_adjusted_power * self.power_scaling_factor
            elif (cluster == 'big') and metric.name in self.big_power_metrics:
                power_metric += metric.value * self.power_scaling_factor
            elif (cluster == 'little') and metric.name in self.little_power_metrics:
                power_metric += metric.value * self.power_scaling_factor
            elif thermal_adjusted_power and metric.name in self.big_energy_metrics:
                power_metric += thermal_adjusted_power / self.run_time * self.power_scaling_factor
            elif (cluster == 'big') and metric.name in self.big_energy_metrics:
                power_metric += metric.value / self.run_time * self.power_scaling_factor
            elif (cluster == 'little') and metric.name in self.little_energy_metrics:
                power_metric += metric.value / self.run_time * self.power_scaling_factor

        if not (power_metric and (perf_metric or not is_freq_iteration)):
            message = 'Incomplete results for {} iteration{}'
            raise InstrumentError(message.format(context.result.spec.id, context.current_iteration))

        if is_freq_iteration:
            index_matter = [cluster, spec.num_cpus,
                            spec.frequency, context.result.iteration]
            data = self.freq_data
        else:
            index_matter = [cluster, spec.num_cpus,
                            spec.idle_state_id, spec.idle_state_desc, context.result.iteration]
            data = self.idle_data
            if self.no_hotplug:
                # due to that fact that hotpluging was disabled, power has to be artificially scaled
                # to the number of cores that should have been active if hotplugging had occurred.
                power_metric = spec.num_cpus * (power_metric / self.number_of_cpus[cluster])

        data.append(index_matter + ['performance', perf_metric])
        data.append(index_matter + ['power', power_metric])

    def before_overall_results_processing(self, context):
        # pylint: disable=too-many-locals
        if not self.idle_data or not self.freq_data:
            self.logger.warning('Run aborted early; not generating energy_model.')
            return
        output_directory = os.path.join(context.output_directory, 'energy_model')
        os.makedirs(output_directory)

        df = pd.DataFrame(self.idle_data, columns=['cluster', 'cpus', 'state_id',
                                                   'state', 'iteration', 'metric', 'value'])
        idle_power_table = wa_result_to_power_perf_table(df, '', index=['cluster', 'cpus', 'state'])
        idle_output = os.path.join(output_directory, IDLE_TABLE_FILE)
        with open(idle_output, 'w') as wfh:
            idle_power_table.to_csv(wfh, index=False)
        context.add_artifact('idle_power_table', idle_output, 'export')

        df = pd.DataFrame(self.freq_data,
                          columns=['cluster', 'cpus', 'frequency', 'iteration', 'metric', 'value'])
        freq_power_table = wa_result_to_power_perf_table(df, self.performance_metric,
                                                         index=['cluster', 'cpus', 'frequency'])
        freq_output = os.path.join(output_directory, FREQ_TABLE_FILE)
        with open(freq_output, 'w') as wfh:
            freq_power_table.to_csv(wfh, index=False)
        context.add_artifact('freq_power_table', freq_output, 'export')

        if self.big_opps is None or self.little_opps is None:
            message = 'OPPs not specified for one or both clusters; cluster power will not be adjusted for leakage.'
            self.logger.warning(message)
        opps = {'big': self.big_opps, 'little': self.little_opps}
        leakages = {'big': self.big_leakage, 'little': self.little_leakage}
        try:
            measured_cpus_table, cpus_table = get_cpus_power_table(freq_power_table, 'frequency', opps, leakages)
        except (ValueError, KeyError, IndexError) as e:
            self.logger.error('Could not create cpu power tables: {}'.format(e))
            return
        measured_cpus_output = os.path.join(output_directory, MEASURED_CPUS_TABLE_FILE)
        with open(measured_cpus_output, 'w') as wfh:
            measured_cpus_table.to_csv(wfh)
        context.add_artifact('measured_cpus_table', measured_cpus_output, 'export')
        cpus_output = os.path.join(output_directory, CPUS_TABLE_FILE)
        with open(cpus_output, 'w') as wfh:
            cpus_table.to_csv(wfh)
        context.add_artifact('cpus_table', cpus_output, 'export')

        em = build_energy_model(freq_power_table, cpus_table, idle_power_table, self.first_cluster_idle_state)
        em_file = os.path.join(output_directory, '{}_em.c'.format(self.device_name))
        em_text = generate_em_c_file(em, self.big_core, self.little_core,
                                     self.em_template_file, em_file)
        context.add_artifact('em', em_file, 'data')

        report_file = os.path.join(output_directory, 'report.html')
        generate_report(freq_power_table, measured_cpus_table, cpus_table,
                        idle_power_table, self.report_template_file,
                        self.device_name, em_text, report_file)
        context.add_artifact('pm_report', report_file, 'export')

    def initialize_result_tracking(self):
        self.freq_data = []
        self.idle_data = []
        self.big_power_metrics = []
        self.little_power_metrics = []
        self.big_energy_metrics = []
        self.little_energy_metrics = []
        if self.power_metric:
            self.big_power_metrics = [pm.format(core=self.big_core) for pm in self.power_metric]
            self.little_power_metrics = [pm.format(core=self.little_core) for pm in self.power_metric]
        else:  # must be energy_metric
            self.big_energy_metrics = [em.format(core=self.big_core) for em in self.energy_metric]
            self.little_energy_metrics = [em.format(core=self.little_core) for em in self.energy_metric]

    def configure_clusters(self):
        self.measured_cores = None
        self.measuring_cores = None
        self.cpuset = self.device.get_cgroup_controller('cpuset')
        self.cpuset.create_group('big', self.big_cpus, [0])
        self.cpuset.create_group('little', self.little_cpus, [0])
        for cluster in set(self.device.core_clusters):
            self.device.set_cluster_governor(cluster, 'userspace')

    def discover_idle_states(self):
        online_cpu = self.device.get_online_cpus(self.big_core)[0]
        self.big_idle_states = self.device.get_cpuidle_states(online_cpu)
        online_cpu = self.device.get_online_cpus(self.little_core)[0]
        self.little_idle_states = self.device.get_cpuidle_states(online_cpu)
        if not (len(self.big_idle_states) >= 2 and len(self.little_idle_states) >= 2):
            raise DeviceError('There do not appeart to be at least two idle states '
                              'on at least one of the clusters.')

    def setup_measurement(self, measured):
        measuring = 'big' if measured == 'little' else 'little'
        self.measured_cluster = measured
        self.measuring_cluster = measuring
        self.measured_cpus = self.big_cpus if measured == 'big' else self.little_cpus
        self.measuring_cpus = self.little_cpus if measured == 'big' else self.big_cpus
        self.reset()

    def reset(self):
        self.enable_all_cores()
        self.enable_all_idle_states()
        self.reset_cgroups()
        self.cpuset.move_all_tasks_to(self.measuring_cluster)
        server_process = 'adbd' if self.device.platform == 'android' else 'sshd'
        server_pids = self.device.get_pids_of(server_process)
        children_ps = [e for e in self.device.ps()
                       if e.ppid in server_pids and e.name != 'sshd']
        children_pids = [e.pid for e in children_ps]
        pids_to_move = server_pids + children_pids
        self.cpuset.root.add_tasks(pids_to_move)
        for pid in pids_to_move:
            try:
                self.device.execute('busybox taskset -p 0x{:x} {}'.format(list_to_mask(self.measuring_cpus), pid))
            except DeviceError:
                pass

    def enable_all_cores(self):
        counter = Counter(self.device.core_names)
        for core, number in counter.iteritems():
            self.device.set_number_of_online_cpus(core, number)
        self.big_cpus = self.device.get_online_cpus(self.big_core)
        self.little_cpus = self.device.get_online_cpus(self.little_core)

    def enable_all_idle_states(self):
        for cpu in self.device.online_cpus:
            for state in self.device.get_cpuidle_states(cpu):
                state.disable = 0

    def reset_cgroups(self):
        self.big_cpus = self.device.get_online_cpus(self.big_core)
        self.little_cpus = self.device.get_online_cpus(self.little_core)
        self.cpuset.big.set(self.big_cpus, 0)
        self.cpuset.little.set(self.little_cpus, 0)

    def perform_runtime_validation(self):
        if not self.device.is_rooted:
            raise InstrumentError('the device must be rooted to generate energy models')
        if 'userspace' not in self.device.list_available_cluster_governors(0):
            raise InstrumentError('userspace cpufreq governor must be enabled')

        error_message = 'Frequency {} is not supported by {} cores'
        available_frequencies = self.device.list_available_core_frequencies(self.big_core)
        if self.big_frequencies:
            for freq in self.big_frequencies:
                if freq not in available_frequencies:
                    raise ConfigError(error_message.format(freq, self.big_core))
        else:
            self.big_frequencies = available_frequencies
        available_frequencies = self.device.list_available_core_frequencies(self.little_core)
        if self.little_frequencies:
            for freq in self.little_frequencies:
                if freq not in available_frequencies:
                    raise ConfigError(error_message.format(freq, self.little_core))
        else:
            self.little_frequencies = available_frequencies

    def initialize_job_queue(self, context):
        old_specs = []
        for job in context.runner.job_queue:
            if job.spec not in old_specs:
                old_specs.append(job.spec)
        new_specs = self.get_cluster_specs(old_specs, 'big', context)
        new_specs.extend(self.get_cluster_specs(old_specs, 'little', context))

        # Update config to refect jobs that will actually run.
        context.config.workload_specs = new_specs
        config_file = os.path.join(context.host_working_directory, 'run_config.json')
        with open(config_file, 'wb') as wfh:
            context.config.serialize(wfh)

        context.runner.init_queue(new_specs)

    def get_cluster_specs(self, old_specs, cluster, context):
        core = self.get_core_name(cluster)
        self.number_of_cpus[cluster] = sum([1 for c in self.device.core_names if c == core])

        cluster_frequencies = self.get_frequencies_param(cluster)
        if not cluster_frequencies:
            raise InstrumentError('Could not read available frequencies for {}'.format(core))
        min_frequency = min(cluster_frequencies)

        idle_states = self.get_device_idle_states(cluster)
        new_specs = []
        for state in idle_states:
            for num_cpus in xrange(1, self.number_of_cpus[cluster] + 1):
                spec = old_specs[0].copy()
                spec.workload_name = self.idle_workload
                spec.workload_parameters = self.idle_workload_params
                spec.idle_state_id = state.id
                spec.idle_state_desc = state.desc
                spec.idle_state_index = state.index
                if not self.no_hotplug:
                    spec.runtime_parameters['{}_cores'.format(core)] = num_cpus
                spec.runtime_parameters['{}_frequency'.format(core)] = min_frequency
                spec.runtime_parameters['ui'] = 'off'
                spec.cluster = cluster
                spec.num_cpus = num_cpus
                spec.id = '{}_idle_{}_{}'.format(cluster, state.id, num_cpus)
                spec.label = 'idle_{}'.format(cluster)
                spec.number_of_iterations = old_specs[0].number_of_iterations
                spec.load(self.device, context.config.ext_loader)
                spec.workload.init_resources(context)
                spec.workload.validate()
                new_specs.append(spec)
        for old_spec in old_specs:
            if old_spec.workload_name not in ['sysbench', 'dhrystone']:
                raise ConfigError('Only sysbench and dhrystone workloads currently supported for energy_model generation.')
            for freq in cluster_frequencies:
                for num_cpus in xrange(1, self.number_of_cpus[cluster] + 1):
                    spec = old_spec.copy()
                    spec.runtime_parameters['{}_frequency'.format(core)] = freq
                    if not self.no_hotplug:
                        spec.runtime_parameters['{}_cores'.format(core)] = num_cpus
                    spec.runtime_parameters['ui'] = 'off'
                    spec.id = '{}_{}_{}'.format(cluster, num_cpus, freq)
                    spec.label = 'freq_{}_{}'.format(cluster, spec.label)
                    spec.workload_parameters['taskset_mask'] = list_to_mask(self.get_cpus(cluster))
                    spec.workload_parameters['threads'] = num_cpus
                    if old_spec.workload_name == 'sysbench':
                        # max_requests set to an arbitrary high values to make sure
                        # sysbench runs for full duriation even on highly
                        # performant cores.
                        spec.workload_parameters['max_requests'] = 10000000
                    spec.cluster = cluster
                    spec.num_cpus = num_cpus
                    spec.frequency = freq
                    spec.load(self.device, context.config.ext_loader)
                    spec.workload.init_resources(context)
                    spec.workload.validate()
                    new_specs.append(spec)
        return new_specs

    def disable_thermal_management(self):
        if self.device.file_exists('/sys/class/thermal/thermal_zone0'):
            tzone_paths = self.device.execute('ls /sys/class/thermal/thermal_zone*')
            for tzpath in tzone_paths.strip().split():
                mode_file = '{}/mode'.format(tzpath)
                if self.device.file_exists(mode_file):
                    self.device.set_sysfile_value(mode_file, 'disabled')

    def get_device_idle_states(self, cluster):
        if cluster == 'big':
            online_cpus = self.device.get_online_cpus(self.big_core)
        else:
            online_cpus = self.device.get_online_cpus(self.little_core)
        idle_states = []
        for cpu in online_cpus:
            idle_states.extend(self.device.get_cpuidle_states(cpu))
        return idle_states

    def get_core_name(self, cluster):
        if cluster == 'big':
            return self.big_core
        else:
            return self.little_core

    def get_cpus(self, cluster):
        if cluster == 'big':
            return self.big_cpus
        else:
            return self.little_cpus

    def get_frequencies_param(self, cluster):
        if cluster == 'big':
            return self.big_frequencies
        else:
            return self.little_frequencies


def _adjust_for_thermal(data, filt_method=lambda x: x, thresh=0.9, window=5000, tdiff_threshold=10000):
    n = filt_method(data)
    n = n[~np.isnan(n)]  # pylint: disable=no-member

    d = np.diff(n)  # pylint: disable=no-member
    d = d[~np.isnan(d)]  # pylint: disable=no-member
    dmin = min(d)
    dmax = max(d)

    index_up = np.max((d > dmax * thresh).nonzero())  # pylint: disable=no-member
    index_down = np.min((d < dmin * thresh).nonzero())  # pylint: disable=no-member
    low_average = np.average(n[index_up:index_up + window])  # pylint: disable=no-member
    high_average = np.average(n[index_down - window:index_down])  # pylint: disable=no-member
    if low_average > high_average or index_down - index_up < tdiff_threshold:
        return 0
    else:
        return low_average


if __name__ == '__main__':
    import sys   # pylint: disable=wrong-import-position,wrong-import-order
    indir, outdir = sys.argv[1], sys.argv[2]
    device_name = 'odroidxu3'
    big_core = 'a15'
    little_core = 'a7'
    first_cluster_idle_state = -1

    this_dir = os.path.dirname(__file__)
    report_template_file = os.path.join(this_dir, REPORT_TEMPLATE_FILE)
    em_template_file = os.path.join(this_dir, EM_TEMPLATE_FILE)

    freq_power_table = pd.read_csv(os.path.join(indir, FREQ_TABLE_FILE))
    measured_cpus_table, cpus_table = pd.read_csv(os.path.join(indir, CPUS_TABLE_FILE),  # pylint: disable=unbalanced-tuple-unpacking
                                                  header=range(2), index_col=0)
    idle_power_table = pd.read_csv(os.path.join(indir, IDLE_TABLE_FILE))

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    report_file = os.path.join(outdir, 'report.html')
    em_file = os.path.join(outdir, '{}_em.c'.format(device_name))

    em = build_energy_model(freq_power_table, cpus_table,
                            idle_power_table, first_cluster_idle_state)
    em_text = generate_em_c_file(em, big_core, little_core,
                                 em_template_file, em_file)
    generate_report(freq_power_table, measured_cpus_table, cpus_table,
                    idle_power_table, report_template_file, device_name,
                    em_text, report_file)
