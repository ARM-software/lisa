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

from __future__ import division
import os

try:
    import pandas as pd
except ImportError:
    pd = None

from past.builtins import basestring

from devlib.derived import DerivedMeasurements, DerivedMetric
from devlib.exception import HostError
from devlib.instrument import MeasurementsCsv
from devlib.utils.csvutil import csvwriter
from devlib.utils.rendering import gfxinfo_get_last_dump, VSYNC_INTERVAL
from devlib.utils.types import numeric


class DerivedFpsStats(DerivedMeasurements):

    def __init__(self, drop_threshold=5, suffix=None, filename=None, outdir=None):
        self.drop_threshold = drop_threshold
        self.suffix = suffix
        self.filename = filename
        self.outdir = outdir
        if (filename is None) and (suffix is None):
            self.suffix = '-fps'
        elif (filename is not None) and (suffix is not None):
            raise ValueError('suffix and filename cannot be specified at the same time.')
        if filename is not None and os.sep in filename:
            raise ValueError('filename cannot be a path (cannot countain "{}"'.format(os.sep))

    # pylint: disable=no-member
    def process(self, measurements_csv):
        if isinstance(measurements_csv, basestring):
            measurements_csv = MeasurementsCsv(measurements_csv)
        if pd is not None:
            return self._process_with_pandas(measurements_csv)
        return self._process_without_pandas(measurements_csv)

    def _get_csv_file_name(self, frames_file):
        outdir = self.outdir or os.path.dirname(frames_file)
        if self.filename:
            return os.path.join(outdir, self.filename)

        frames_basename = os.path.basename(frames_file)
        rest, ext = os.path.splitext(frames_basename)
        csv_basename = rest + self.suffix + ext
        return os.path.join(outdir, csv_basename)


class DerivedGfxInfoStats(DerivedFpsStats):

    #pylint: disable=arguments-differ
    @staticmethod
    def process_raw(filepath, *args):
        metrics = []
        dump = gfxinfo_get_last_dump(filepath)
        seen_stats = False
        for line in dump.split('\n'):
            if seen_stats and not line.strip():
                break
            elif line.startswith('Janky frames:'):
                text = line.split(': ')[-1]
                val_text, pc_text = text.split('(')
                metrics.append(DerivedMetric('janks', numeric(val_text.strip()), 'count'))
                metrics.append(DerivedMetric('janks_pc', numeric(pc_text[:-3]), 'percent'))
            elif ' percentile: ' in line:
                ptile, val_text = line.split(' percentile: ')
                name = 'render_time_{}_ptile'.format(ptile)
                value = numeric(val_text.strip()[:-2])
                metrics.append(DerivedMetric(name, value, 'time_ms'))
            elif line.startswith('Number '):
                name_text, val_text = line.strip().split(': ')
                name = name_text[7:].lower().replace(' ', '_')
                value = numeric(val_text)
                metrics.append(DerivedMetric(name, value, 'count'))
            else:
                continue
            seen_stats = True
        return metrics

    def _process_without_pandas(self, measurements_csv):
        per_frame_fps = []
        start_vsync, end_vsync = None, None
        frame_count = 0

        for frame_data in measurements_csv.iter_values():
            if frame_data.Flags_flags != 0:
                continue
            frame_count += 1

            if start_vsync is None:
                start_vsync = frame_data.Vsync_time_ns
            end_vsync = frame_data.Vsync_time_ns

            frame_time = frame_data.FrameCompleted_time_ns - frame_data.IntendedVsync_time_ns
            pff = 1e9 / frame_time
            if pff > self.drop_threshold:
                per_frame_fps.append([pff])

        if frame_count:
            duration = end_vsync - start_vsync
            fps = (1e9 * frame_count) / float(duration)
        else:
            duration = 0
            fps = 0

        csv_file = self._get_csv_file_name(measurements_csv.path)
        with csvwriter(csv_file) as writer:
            writer.writerow(['fps'])
            writer.writerows(per_frame_fps)

        return [DerivedMetric('fps', fps, 'fps'),
                DerivedMetric('total_frames', frame_count, 'frames'),
                MeasurementsCsv(csv_file)]

    def _process_with_pandas(self, measurements_csv):
        data = pd.read_csv(measurements_csv.path)
        data = data[data.Flags_flags == 0]
        frame_time = data.FrameCompleted_time_ns - data.IntendedVsync_time_ns
        per_frame_fps = (1e9 / frame_time)
        keep_filter = per_frame_fps > self.drop_threshold
        per_frame_fps = per_frame_fps[keep_filter]
        per_frame_fps.name = 'fps'

        frame_count = data.index.size
        if frame_count > 1:
            duration = data.Vsync_time_ns.iloc[-1] - data.Vsync_time_ns.iloc[0]
            fps = (1e9 * frame_count) / float(duration)
        else:
            duration = 0
            fps = 0

        csv_file = self._get_csv_file_name(measurements_csv.path)
        per_frame_fps.to_csv(csv_file, index=False, header=True)

        return [DerivedMetric('fps', fps, 'fps'),
                DerivedMetric('total_frames', frame_count, 'frames'),
                MeasurementsCsv(csv_file)]


class DerivedSurfaceFlingerStats(DerivedFpsStats):

    # pylint: disable=too-many-locals
    def _process_with_pandas(self, measurements_csv):
        data = pd.read_csv(measurements_csv.path)

        # fiter out bogus frames.
        bogus_frames_filter = data.actual_present_time_us != 0x7fffffffffffffff
        actual_present_times = data.actual_present_time_us[bogus_frames_filter]
        actual_present_time_deltas = actual_present_times.diff().dropna()

        vsyncs_to_compose = actual_present_time_deltas.div(VSYNC_INTERVAL)
        vsyncs_to_compose.apply(lambda x: int(round(x, 0)))

        # drop values lower than drop_threshold FPS as real in-game frame
        # rate is unlikely to drop below that (except on loading screens
        # etc, which should not be factored in frame rate calculation).
        per_frame_fps = (1.0 / (vsyncs_to_compose.multiply(VSYNC_INTERVAL / 1e9)))
        keep_filter = per_frame_fps > self.drop_threshold
        filtered_vsyncs_to_compose = vsyncs_to_compose[keep_filter]
        per_frame_fps.name = 'fps'

        csv_file = self._get_csv_file_name(measurements_csv.path)
        per_frame_fps.to_csv(csv_file, index=False, header=True)

        if not filtered_vsyncs_to_compose.empty:
            fps = 0
            total_vsyncs = filtered_vsyncs_to_compose.sum()
            frame_count = filtered_vsyncs_to_compose.size

            if total_vsyncs:
                fps = 1e9 * frame_count / (VSYNC_INTERVAL * total_vsyncs)

            janks = self._calc_janks(filtered_vsyncs_to_compose)
            not_at_vsync = self._calc_not_at_vsync(vsyncs_to_compose)
        else:
            fps = 0
            frame_count = 0
            janks = 0
            not_at_vsync = 0

        janks_pc = 0 if frame_count == 0 else janks * 100 / frame_count

        return [DerivedMetric('fps', fps, 'fps'),
                DerivedMetric('total_frames', frame_count, 'frames'),
                MeasurementsCsv(csv_file),
                DerivedMetric('janks', janks, 'count'),
                DerivedMetric('janks_pc', janks_pc, 'percent'),
                DerivedMetric('missed_vsync', not_at_vsync, 'count')]

    # pylint: disable=unused-argument,no-self-use
    def _process_without_pandas(self, measurements_csv):
        # Given that SurfaceFlinger has been deprecated in favor of GfxInfo,
        # it does not seem worth it implementing this.
        raise HostError('Please install "pandas" Python package to process SurfaceFlinger frames')

    @staticmethod
    def _calc_janks(filtered_vsyncs_to_compose):
        """
        Internal method for calculating jank frames.
        """
        pause_latency = 20
        vtc_deltas = filtered_vsyncs_to_compose.diff().dropna()
        vtc_deltas = vtc_deltas.abs()
        janks = vtc_deltas.apply(lambda x: (pause_latency > x > 1.5) and 1 or 0).sum()

        return janks

    @staticmethod
    def _calc_not_at_vsync(vsyncs_to_compose):
        """
        Internal method for calculating the number of frames that did not
        render in a single vsync cycle.
        """
        epsilon = 0.0001
        func = lambda x: (abs(x - 1.0) > epsilon) and 1 or 0
        not_at_vsync = vsyncs_to_compose.apply(func).sum()

        return not_at_vsync
