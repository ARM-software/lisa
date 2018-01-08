
from wa import OutputProcessor
from wa.utils.android import LogcatParser


class UxperfProcessor(OutputProcessor):

    name = 'uxperf'

    description = '''
    Parse logcat for UX_PERF markers to produce performance metrics for
    workload actions using specified instrumentation.
    An action represents a series of UI interactions to capture.
    NOTE: The UX_PERF markers are turned off by default and must be enabled in
    a agenda file by setting ``markers_enabled`` for the workload to ``True``.
    '''

    def process_job_output(self, output, target_info, job_output):
        logcat = output.get_artifact('logcat')
        if not logcat:
            return

        parser = LogcatParser()
        start_times = {}

        filepath = output.get_path(logcat.path)
        for entry in parser.parse(filepath):
            if not entry.tag == 'UX_PERF':
                continue

            parts = entry.message.split()
            if len(parts) != 3:
                message = 'Unexpected UX_PERF message @ {}: {}'
                self.logger.warning(message.format(entry.timestamp, entry.message))
                continue

            action, state, when = parts
            when = int(when)
            if state == 'start':
                if action in start_times:
                    self.logger.warning('start before end @ {}'.format(entry.timestamp))
                start_times[action] = when
            elif state == 'end':
                start_time = start_times.pop(action, None)
                if start_time is None:
                    self.logger.warning('end without start @ {}'.format(entry.timestamp))
                    continue

                duration = (when - start_time) / 1000
                metric_name = '{}_duration'.format(action)
                output.add_metric(metric_name, duration, 'microseconds',
                                  lower_is_better=True)

            else:
                self.logger.warning('Unexpected state "{}" @ {}'.format(state, entry.timestamp))
