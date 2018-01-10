import os
import shutil

from devlib import SurfaceFlingerFramesInstrument, GfxInfoFramesInstrument
from devlib import DerivedSurfaceFlingerStats, DerivedGfxInfoStats

from wa import Instrument, Parameter, WorkloadError
from wa.utils.types import numeric


class FpsInstrument(Instrument):

    name = 'fps'
    description = """
    Measures Frames Per Second (FPS) and associated metrics for a workload.

    .. note:: This instrument depends on pandas Python library (which is not part of standard
              WA dependencies), so you will need to install that first, before you can use it.

    Android L and below use SurfaceFlinger to calculate the FPS data.
    Android M and above use gfxinfo to calculate the FPS data.

    SurfaceFlinger:
    The view is specified by the workload as ``view`` attribute. This defaults
    to ``'SurfaceView'`` for game workloads, and ``None`` for non-game
    workloads (as for them FPS mesurement usually doesn't make sense).
    Individual workloads may override this.

    gfxinfo:
    The view is specified by the workload as ``package`` attribute.
    This is because gfxinfo already processes for all views in a package.

    """

    parameters = [
        Parameter('drop_threshold', kind=numeric, default=5,
                  description="""
                  Data points below this FPS will be dropped as they do not
                  constitute "real" gameplay. The assumption being that while
                  actually running, the FPS in the game will not drop below X
                  frames per second, except on loading screens, menus, etc,
                  which should not contribute to FPS calculation.
                  """),
        Parameter('keep_raw', kind=bool, default=False,
                  description="""
                  If set to ``True``, this will keep the raw dumpsys output in
                  the results directory (this is maily used for debugging)
                  Note: frames.csv with collected frames data will always be
                  generated regardless of this setting.
                   """),
        Parameter('crash_threshold', kind=float, default=0.7,
                  description="""
                  Specifies the threshold used to decided whether a
                  measured/expected frames ration indicates a content crash.
                  E.g. a value of ``0.75`` means the number of actual frames
                  counted is a quarter lower than expected, it will treated as
                  a content crash.

                  If set to zero, no crash check will be performed.
                  """),
        Parameter('period', kind=float, default=2, constraint=lambda x: x > 0,
                  description="""
                  Specifies the time period between polling frame data in
                  seconds when collecting frame data. Using a lower value
                  improves the granularity of timings when recording actions
                  that take a short time to complete.  Note, this will produce
                  duplicate frame data in the raw dumpsys output, however, this
                  is filtered out in frames.csv.  It may also affect the
                  overall load on the system.

                  The default value of 2 seconds corresponds with the
                  NUM_FRAME_RECORDS in
                  android/services/surfaceflinger/FrameTracker.h (as of the
                  time of writing currently 128) and a frame rate of 60 fps
                  that is applicable to most devices.
                  """),
        Parameter('force_surfaceflinger', kind=bool, default=False,
                  description="""
                  By default, the method to capture fps data is based on
                  Android version.  If this is set to true, force the
                  instrument to use the SurfaceFlinger method regardless of its
                  Android version.
                  """),
    ]

    def __init__(self, target, **kwargs):
        super(FpsInstrument, self).__init__(target, **kwargs)
        self.collector = None
        self.processor = None
        self._is_enabled = None

    def setup(self, context):
        use_gfxinfo = self.target.get_sdk_version() >= 23 and not self.force_surfaceflinger
        if use_gfxinfo:
            collector_target_attr = 'package'
        else:
            collector_target_attr = 'view'
        collector_target = getattr(context.workload, collector_target_attr, None)

        if not collector_target:
            self._is_enabled = False
            msg = 'Workload {} does not define a {}; disabling frame collection and FPS evaluation.'
            self.logger.info(msg.format(context.workload.name, collector_target_attr))
            return

        self._is_enabled = True
        if use_gfxinfo:
            self.collector = GfxInfoFramesInstrument(self.target, collector_target, self.period)
            self.processor = DerivedGfxInfoStats(self.drop_threshold, filename='fps.csv')
        else:
            self.collector = SurfaceFlingerFramesInstrument(self.target, collector_target, self.period)
            self.processor = DerivedSurfaceFlingerStats(self.drop_threshold, filename='fps.csv')
        self.collector.reset()

    def start(self, context):
        if not self._is_enabled:
            return
        self.collector.start()

    def stop(self, context):
        if not self._is_enabled:
            return
        self.collector.stop()

    def update_output(self, context):
        if not self._is_enabled:
            return
        outpath = os.path.join(context.output_directory, 'frames.csv')
        frames_csv = self.collector.get_data(outpath)
        raw_output = self.collector.get_raw()

        processed = self.processor.process(frames_csv)
        processed.extend(self.processor.process_raw(*raw_output))
        fps, frame_count, fps_csv = processed[:3]
        rest = processed[3:]

        context.add_metric(fps.name, fps.value, fps.units)
        context.add_metric(frame_count.name, frame_count.value, frame_count.units)
        context.add_artifact('frames', frames_csv.path, kind='raw')
        context.add_artifact('fps', fps_csv.path, kind='data')
        for metric in rest:
            context.add_metric(metric.name, metric.value, metric.units, lower_is_better=True)

        if not self.keep_raw:
            for entry in raw_output:
                if os.path.isdir(entry):
                    shutil.rmtree(entry)
                elif os.path.isfile(entry):
                    os.remove(entry)

        if not frame_count.value:
            context.add_event('Could not frind frames data in gfxinfo output')
            context.set_status('PARTIAL')

        self.check_for_crash(context, fps.value, frame_count.value,
                             context.current_job.run_time.total_seconds())

    def check_for_crash(self, context, fps, frames, exec_time):
        if not self.crash_threshold:
            return
        self.logger.debug('Checking for crashed content.')
        if all([exec_time, fps, frames]):
            expected_frames = fps * exec_time
            ratio = frames / expected_frames
            self.logger.debug('actual/expected frames: {:.2}'.format(ratio))
            if ratio < self.crash_threshold:
                msg = 'Content for {} appears to have crashed.\n'.format(context.current_job.spec.label)
                msg += 'Content crash detected (actual/expected frames: {:.2}).'.format(ratio)
                raise WorkloadError(msg)
