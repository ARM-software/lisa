#    Copyright 2013-2015 ARM Limited
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


# pylint: disable=W0613,E1101
from __future__ import division
import os
import sys
import time
import csv
import shutil
import threading
import errno
import tempfile

from distutils.version import LooseVersion


from wlauto import Instrument, Parameter, IterationResult
from wlauto.instrumentation import instrument_is_installed
from wlauto.exceptions import (InstrumentError, WorkerThreadError, ConfigError,
                               DeviceNotRespondingError, TimeoutError)
from wlauto.utils.types import boolean, numeric

try:
    import pandas as pd
except ImportError:
    pd = None


VSYNC_INTERVAL = 16666667
PAUSE_LATENCY = 20
EPSYLON = 0.0001


class FpsInstrument(Instrument):

    name = 'fps'
    description = """
    Measures Frames Per Second (FPS) and associated metrics for a workload's main View.

    .. note:: This instrument depends on pandas Python library (which is not part of standard
              WA dependencies), so you will need to install that first, before you can use it.

    The view is specified by the workload as ``view`` attribute. This defaults
    to ``'SurfaceView'`` for game workloads, and ``None`` for non-game
    workloads (as for them FPS mesurement usually doesn't make sense).
    Individual workloads may override this.

    This instrument adds four metrics to the results:

        :FPS: Frames Per Second. This is the frame rate of the workload.
        :frames: The total number of frames rendered during the execution of
                 the workload.
        :janks: The number of "janks" that occured during execution of the
                workload. Janks are sudden shifts in frame rate. They result
                in a "stuttery" UI. See http://jankfree.org/jank-busters-io
        :not_at_vsync: The number of frames that did not render in a single
                       vsync cycle.

    """
    supported_platforms = ['android']

    parameters = [
        Parameter('drop_threshold', kind=numeric, default=5,
                  description='Data points below this FPS will be dropped as they '
                              'do not constitute "real" gameplay. The assumption '
                              'being that while actually running, the FPS in the '
                              'game will not drop below X frames per second, '
                              'except on loading screens, menus, etc, which '
                              'should not contribute to FPS calculation. '),
        Parameter('keep_raw', kind=boolean, default=False,
                  description='If set to ``True``, this will keep the raw dumpsys output '
                              'in the results directory (this is maily used for debugging) '
                              'Note: frames.csv with collected frames data will always be '
                              'generated regardless of this setting.'),
        Parameter('generate_csv', kind=boolean, default=True,
                  description='If set to ``True``, this will produce temporal fps data '
                              'in the results directory, in a file named fps.csv '
                              'Note: fps data will appear as discrete step-like values '
                              'in order to produce a more meainingfull representation,'
                              'a rolling mean can be applied.'),
        Parameter('crash_check', kind=boolean, default=True,
                  description="""
                  Specifies wither the instrument should check for crashed content by examining
                  frame data. If this is set, ``execution_time`` instrument must also be installed.
                  The check is performed by using the measured FPS and exection time to estimate the expected
                  frames cound and comparing that against the measured frames count. The the ratio of
                  measured/expected is too low, then it is assumed that the content has crashed part way
                  during the run. What is "too low" is determined by ``crash_threshold``.

                  .. note:: This is not 100\% fool-proof. If the crash occurs sufficiently close to
                            workload's termination,  it may not be detected. If this is expected, the
                            threshold may be adjusted up to compensate.
                  """),
        Parameter('crash_threshold', kind=float, default=0.7,
                  description="""
                  Specifies the threshold used to decided whether a measured/expected frames ration indicates
                  a content crash. E.g. a value of ``0.75`` means the number of actual frames counted is a
                  quarter lower than expected, it will treated as a content crash.
                  """),
    ]

    clear_command = 'dumpsys SurfaceFlinger --latency-clear '

    def __init__(self, device, **kwargs):
        super(FpsInstrument, self).__init__(device, **kwargs)
        self.collector = None
        self.outfile = None
        self.fps_outfile = None
        self.is_enabled = True

    def validate(self):
        if not pd or LooseVersion(pd.__version__) < LooseVersion('0.13.1'):
            message = ('fps instrument requires pandas Python package (version 0.13.1 or higher) to be installed.\n'
                       'You can install it with pip, e.g. "sudo pip install pandas"')
            raise InstrumentError(message)
        if self.crash_check and not instrument_is_installed('execution_time'):
            raise ConfigError('execution_time instrument must be installed in order to check for content crash.')

    def setup(self, context):
        workload = context.workload
        if hasattr(workload, 'view'):
            self.fps_outfile = os.path.join(context.output_directory, 'fps.csv')
            self.outfile = os.path.join(context.output_directory, 'frames.csv')
            self.collector = LatencyCollector(self.outfile, self.device, workload.view or '', self.keep_raw, self.logger)
            self.device.execute(self.clear_command)
        else:
            self.logger.debug('Workload does not contain a view; disabling...')
            self.is_enabled = False

    def start(self, context):
        if self.is_enabled:
            self.logger.debug('Starting SurfaceFlinger collection...')
            self.collector.start()

    def stop(self, context):
        if self.is_enabled and self.collector.is_alive():
            self.logger.debug('Stopping SurfaceFlinger collection...')
            self.collector.stop()

    def update_result(self, context):
        if self.is_enabled:
            data = pd.read_csv(self.outfile)
            if not data.empty:  # pylint: disable=maybe-no-member
                per_frame_fps = self._update_stats(context, data)
                if self.generate_csv:
                    per_frame_fps.to_csv(self.fps_outfile, index=False, header=True)
                    context.add_artifact('fps', path='fps.csv', kind='data')
            else:
                context.result.add_metric('FPS', float('nan'))
                context.result.add_metric('frame_count', 0)
                context.result.add_metric('janks', 0)
                context.result.add_metric('not_at_vsync', 0)

    def slow_update_result(self, context):
        result = context.result
        if result.has_metric('execution_time'):
            self.logger.debug('Checking for crashed content.')
            exec_time = result['execution_time'].value
            fps = result['FPS'].value
            frames = result['frame_count'].value
            if all([exec_time, fps, frames]):
                expected_frames = fps * exec_time
                ratio = frames / expected_frames
                self.logger.debug('actual/expected frames: {:.2}'.format(ratio))
                if ratio < self.crash_threshold:
                    self.logger.error('Content for {} appears to have crashed.'.format(context.spec.label))
                    result.status = IterationResult.FAILED
                    result.add_event('Content crash detected (actual/expected frames: {:.2}).'.format(ratio))

    def _update_stats(self, context, data):  # pylint: disable=too-many-locals
        vsync_interval = self.collector.refresh_period
        # fiter out bogus frames.
        actual_present_times = data.actual_present_time[data.actual_present_time != 0x7fffffffffffffff]
        actual_present_time_deltas = (actual_present_times - actual_present_times.shift()).drop(0)  # pylint: disable=E1103
        vsyncs_to_compose = (actual_present_time_deltas / vsync_interval).apply(lambda x: int(round(x, 0)))
        # drop values lower than drop_threshold FPS as real in-game frame
        # rate is unlikely to drop below that (except on loading screens
        # etc, which should not be factored in frame rate calculation).
        per_frame_fps = (1.0 / (vsyncs_to_compose * (vsync_interval / 1e9)))
        keep_filter = per_frame_fps > self.drop_threshold
        filtered_vsyncs_to_compose = vsyncs_to_compose[keep_filter]
        if not filtered_vsyncs_to_compose.empty:
            total_vsyncs = filtered_vsyncs_to_compose.sum()
            if total_vsyncs:
                frame_count = filtered_vsyncs_to_compose.size
                fps = 1e9 * frame_count / (vsync_interval * total_vsyncs)
                context.result.add_metric('FPS', fps)
                context.result.add_metric('frame_count', frame_count)
            else:
                context.result.add_metric('FPS', float('nan'))
                context.result.add_metric('frame_count', 0)

            vtc_deltas = filtered_vsyncs_to_compose - filtered_vsyncs_to_compose.shift()
            vtc_deltas.index = range(0, vtc_deltas.size)
            vtc_deltas = vtc_deltas.drop(0).abs()
            janks = vtc_deltas.apply(lambda x: (PAUSE_LATENCY > x > 1.5) and 1 or 0).sum()
            not_at_vsync = vsyncs_to_compose.apply(lambda x: (abs(x - 1.0) > EPSYLON) and 1 or 0).sum()
            context.result.add_metric('janks', janks)
            context.result.add_metric('not_at_vsync', not_at_vsync)
        else:  # no filtered_vsyncs_to_compose
            context.result.add_metric('FPS', float('nan'))
            context.result.add_metric('frame_count', 0)
            context.result.add_metric('janks', 0)
            context.result.add_metric('not_at_vsync', 0)
        per_frame_fps.name = 'fps'
        return per_frame_fps


class LatencyCollector(threading.Thread):

    # Note: the size of the frames buffer for a particular surface is defined
    #       by NUM_FRAME_RECORDS inside android/services/surfaceflinger/FrameTracker.h.
    #       At the time of writing, this was hard-coded to 128. So at 60 fps
    #       (and there is no reason to go above that, as it matches vsync rate
    #       on pretty much all phones), there is just over 2 seconds' worth of
    #       frames in there. Hence the sleep time of 2 seconds between dumps.
    #command_template = 'while (true); do dumpsys SurfaceFlinger --latency {}; sleep 2; done'
    command_template = 'dumpsys SurfaceFlinger --latency {}'

    def __init__(self, outfile, device, activity, keep_raw, logger):
        super(LatencyCollector, self).__init__()
        self.outfile = outfile
        self.device = device
        self.command = self.command_template.format(activity)
        self.keep_raw = keep_raw
        self.logger = logger
        self.stop_signal = threading.Event()
        self.frames = []
        self.last_ready_time = 0
        self.refresh_period = VSYNC_INTERVAL
        self.drop_threshold = self.refresh_period * 1000
        self.exc = None
        self.unresponsive_count = 0

    def run(self):
        try:
            self.logger.debug('SurfaceFlinger collection started.')
            self.stop_signal.clear()
            fd, temp_file = tempfile.mkstemp()
            self.logger.debug('temp file: {}'.format(temp_file))
            wfh = os.fdopen(fd, 'wb')
            try:
                while not self.stop_signal.is_set():
                    wfh.write(self.device.execute(self.command))
                    time.sleep(2)
            finally:
                wfh.close()
            # TODO: this can happen after the run during results processing
            with open(temp_file) as fh:
                text = fh.read().replace('\r\n', '\n').replace('\r', '\n')
                for line in text.split('\n'):
                    line = line.strip()
                    if line:
                        self._process_trace_line(line)
            if self.keep_raw:
                raw_file = os.path.join(os.path.dirname(self.outfile), 'surfaceflinger.raw')
                shutil.copy(temp_file, raw_file)
            os.unlink(temp_file)
        except (DeviceNotRespondingError, TimeoutError):  # pylint: disable=W0703
            raise
        except Exception, e:  # pylint: disable=W0703
            self.logger.warning('Exception on collector thread: {}({})'.format(e.__class__.__name__, e))
            self.exc = WorkerThreadError(self.name, sys.exc_info())
        self.logger.debug('SurfaceFlinger collection stopped.')

        with open(self.outfile, 'w') as wfh:
            writer = csv.writer(wfh)
            writer.writerow(['desired_present_time', 'actual_present_time', 'frame_ready_time'])
            writer.writerows(self.frames)
        self.logger.debug('Frames data written.')

    def stop(self):
        self.stop_signal.set()
        self.join()
        if self.unresponsive_count:
            message = 'SurfaceFlinger was unrepsonsive {} times.'.format(self.unresponsive_count)
            if self.unresponsive_count > 10:
                self.logger.warning(message)
            else:
                self.logger.debug(message)
        if self.exc:
            raise self.exc  # pylint: disable=E0702
        self.logger.debug('FSP collection complete.')

    def _process_trace_line(self, line):
        parts = line.split()
        if len(parts) == 3:
            desired_present_time, actual_present_time, frame_ready_time = map(int, parts)
            if frame_ready_time <= self.last_ready_time:
                return  # duplicate frame
            if (frame_ready_time - desired_present_time) > self.drop_threshold:
                self.logger.debug('Dropping bogus frame {}.'.format(line))
                return  # bogus data
            self.last_ready_time = frame_ready_time
            self.frames.append((desired_present_time, actual_present_time, frame_ready_time))
        elif len(parts) == 1:
            self.refresh_period = int(parts[0])
            self.drop_threshold = self.refresh_period * 10
        elif 'SurfaceFlinger appears to be unresponsive, dumping anyways' in line:
            self.unresponsive_count += 1
        else:
            self.logger.warning('Unexpected SurfaceFlinger dump output: {}'.format(line))
