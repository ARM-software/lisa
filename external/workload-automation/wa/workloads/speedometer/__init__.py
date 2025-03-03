#    Copyright 2014-2018 ARM Limited
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
from collections import defaultdict
from http.server import SimpleHTTPRequestHandler, HTTPServer
import logging
import os
import re
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid

from wa import Parameter, Workload, File
from wa.framework.exception import WorkloadError
from wa.utils.exec_control import once
from wa.utils.misc import safe_extract

from devlib.utils.android import adb_command


class Speedometer(Workload):

    name = "speedometer"
    description = """
    A workload to execute the speedometer 2.0 web based benchmark. Requires device to be rooted.
    This workload will only with Android 9+ devices if connected via TCP, or Android 5+ if
    connected via USB.

    Test description:

    1. Host a local copy of the Speedometer website, and make it visible to the device via ADB.
    2. Open chrome via an intent to access the local copy.
    3. Execute the benchmark - the copy has been modified to trigger the start of the benchmark.
    4. The benchmark will write to the browser's sandboxed local storage to signal the benchmark
       has completed. This local storage is monitored by this workload.

    Known working chrome version 83.0.4103.106

    To modify the archived speedometer workload:

    1. Run 'git clone https://github.com/WebKit/webkit'

    2. Copy PerformanceTests/Speedometer to a directory called document_root, renaming Speedometer to Speedometer2.0

    3. Modify document_root/Speedometer2.0/index.html:

      3a. Remove the 'defer' attribute from the <script> tags within the <head> section.
      3b. Add '<script>startTest();</script>' to the very end of the <body> section.

    4. Modify document_root/Speedometer2.0/resources/main.js:

      4a. Add the listed code after this line:

            document.getElementById('result-number').textContent = results.formattedMean;

        Code to add::

                if (location.search.length > 1) {
                    var parts = location.search.substring(1).split('&');
                    for (var i = 0; i < parts.length; i++) {
                        var keyValue = parts[i].split('=');
                        var key = keyValue[0];
                        var value = keyValue[1];
                        if (key === "reportEndId") {
                            window.localStorage.setItem('reportEndId', value);
                        }
                    }
                }

    5. Run 'tar -cpzf speedometer_archive.tgz document_root'

    6. Copy the tarball into the workloads/speedometer directory

    7. If appropriate, update the commit info in the LICENSE file.
    """
    supported_platforms = ["android"]

    package_names = ["org.chromium.chrome", "com.android.chrome", "org.bromite.chromium"]
    # This regex finds a single XML tag where property 1 and 2 are true:
    #  1. contains the attribute text="XXX" or content-desc="XXX"
    #  2. and exclusively either 2a or 2b is true:
    #   2a. there exists a index="3" or resource-id="result-number" to that attribute's left
    #   2b. there exists a resource-id="result-number" to that attribute's right
    # The regex stores the XXX value of that attribute in the named group 'value'.
    #
    # Just in case someone wants to learn something:
    #  If you use (?P<tag>regex)? to match 'regex', and then afterwards you
    #  have (?(tag)A|B), then regex A will be used if the 'tag' group captured
    #  something and B will be used if nothing was captured. This is how we
    #  search for only 'resource-id="result-number"' after the text/content-desc
    #  _only_ in the case we didn't see it before.
    #  Since 'index="3"' is always on the left side of the value.
    regex = re.compile(
        r'<[^>]*(?P<Z>index="3"|resource-id="result-number")?[^>]*'
        r'(?:text|content-desc)="(?P<value>\d+.\d+)"[^>]*'
        r'(?(Z)|resource-id="result-number")[^>]*\/>'
    )

    parameters = [
        Parameter(
            "chrome_package",
            allowed_values=package_names,
            kind=str,
            default="com.android.chrome",
            description="""
                  The app package for the browser that will be launched.
                  """,
        ),
    ]

    def __init__(self, target, **kwargs):
        super(Speedometer, self).__init__(target, **kwargs)
        self.target_file_was_seen = defaultdict(lambda: False)
        self.ui_dump_loc = None

    @once
    def initialize(self, context):
        super(Speedometer, self).initialize(context)
        Speedometer.archive_server = ArchiveServer()
        if not self.target.is_rooted:
            raise WorkloadError(
                "Device must be rooted for the speedometer workload currently"
            )

        if not self.target.package_is_installed(self.chrome_package):
            raise WorkloadError(
                "Could not find '{}' on the device. Please ensure it is installed, "
                "or specify the correct package name using 'chrome_package' "
                "parameter.".format(self.chrome_package))

        if self.target.adb_server is not None:
            raise WorkloadError(
                "Workload does not support the adb_server parameter, due to the webpage "
                "hosting mechanism."
            )

        # Temporary directory used for storing the Speedometer files, uiautomator
        # dumps, and modified XML chrome config files.
        Speedometer.temp_dir = tempfile.TemporaryDirectory()
        Speedometer.document_root = os.path.join(self.temp_dir.name, "document_root")

        # Host a copy of Speedometer locally
        tarball = context.get_resource(File(self, "speedometer_archive.tgz"))
        with tarfile.open(name=tarball) as handle:
            safe_extract(handle, self.temp_dir.name)
        self.archive_server.start(self.document_root)

        Speedometer.speedometer_url = "http://localhost:{}/Speedometer2.0/index.html".format(
            self.archive_server.get_port()
        )

    def setup(self, context):
        super(Speedometer, self).setup(context)

        # We are making sure we start with a 'fresh' browser - no other tabs,
        # nothing in the page cache, etc.

        # Clear the application's cache.
        self.target.execute("pm clear {}".format(self.chrome_package), as_root=True)

        # Launch the browser for the first time and then stop it. Since the
        # cache has just been cleared, this forces it to recreate its
        # preferences file, that we need to modify.
        browser_launch_cmd = "am start -a android.intent.action.VIEW -d {} {}".format(
            self.speedometer_url, self.chrome_package
        )
        self.target.execute(browser_launch_cmd)
        time.sleep(1)
        self.target.execute("am force-stop {}".format(self.chrome_package))
        time.sleep(1)

        # Pull the preferences file from the device, modify it, and push it
        # back.  This is done to bypass the 'first launch' screen of the
        # browser we see after the cache is cleared.
        self.preferences_xml = "{}_preferences.xml".format(self.chrome_package)

        file_to_modify = "/data/data/{}/shared_prefs/{}".format(
            self.chrome_package, self.preferences_xml
        )

        self.target.pull(file_to_modify, self.temp_dir.name, as_root=True)

        with open(os.path.join(self.temp_dir.name, self.preferences_xml)) as read_fh:
            lines = read_fh.readlines()

            # Add additional elements for the preferences XML to the
            # _second-last_ line
            for line in [
                '<boolean name="first_run_flow" value="true" />\n',
                '<boolean name="first_run_tos_accepted" value="true" />\n',
                '<boolean name="first_run_signin_complete" value="true" />\n',
                '<boolean name="displayed_data_reduction_promo" value="true" />\n',
                # Add a 'request count' value to dismiss the pop-up window on the screen.
                # If the value is greater than 1, pop-up window will be dismissed.
                '<int name="Chrome.NotificationPermission.RequestCount" value="2" />\n',
            ]:
                lines.insert(len(lines) - 1, line)

            with open(
                os.path.join(self.temp_dir.name, self.preferences_xml + ".new"), "w",
            ) as write_fh:
                for line in lines:
                    write_fh.write(line)

        # Make sure ownership of the original file is preserved.
        user_owner, group_owner = self.target.execute(
            "ls -l {}".format(file_to_modify), as_root=True,
        ).split()[2:4]

        self.target.push(
            os.path.join(self.temp_dir.name, self.preferences_xml + ".new"),
            file_to_modify,
            as_root=True,
        )

        self.target.execute(
            "chown {}.{} {}".format(user_owner, group_owner, file_to_modify),
            as_root=True,
        )

    def run(self, context):
        super(Speedometer, self).run(context)

        self.archive_server.expose_to_device(self.target)

        # Generate a UUID to search for in the browser's local storage to find out
        # when the workload has ended.
        report_end_id = uuid.uuid4().hex
        url_with_unique_id = "{}?reportEndId={}".format(
            self.speedometer_url, report_end_id
        )

        browser_launch_cmd = "am start -a android.intent.action.VIEW -d '{}' {}".format(
            url_with_unique_id, self.chrome_package
        )
        self.target.execute(browser_launch_cmd)

        self.wait_for_benchmark_to_complete(report_end_id)

        self.archive_server.hide_from_device(self.target)

    def target_file_was_created(self, f):
        """Assume that once self.target.file_exists(f) returns True, it will
        always be True from that point forward, so cache the response into the
        self.target_file_was_seen dict."""
        if not self.target_file_was_seen[f]:
            self.target_file_was_seen[f] = self.target.file_exists(f)
        return self.target_file_was_seen[f]

    def wait_for_benchmark_to_complete(self, report_end_id):
        local_storage = "/data/data/{}/app_chrome/Default/Local Storage/leveldb".format(
            self.chrome_package
        )

        sleep_period_s = 5
        find_period_s = 30
        timeout_period_m = 15

        iterations = 0
        local_storage_seen = False
        benchmark_complete = False
        while not benchmark_complete:
            if self.target_file_was_created(local_storage):
                if (
                    iterations % (find_period_s // sleep_period_s) == 0
                    or not local_storage_seen
                ):
                    # There's a chance we don't see the localstorage file immediately, and there's a
                    # chance more of them could be created later, so check for those files every ~30
                    # seconds.
                    find_cmd = '{} find "{}" -iname "*.log"'.format(
                        self.target.busybox, local_storage
                    )
                    candidate_files = self.target.execute(find_cmd, as_root=True).split(
                        "\n"
                    )

                local_storage_seen = True

                for ls_file in candidate_files:
                    # Each local storage file is in a binary format. The busybox grep seems to
                    # print out the line '[KEY][VALUE]' for a match, rather than just reporting
                    # that 'binary file X matches', so just check the output for our generated ID.
                    grep_cmd = '{} grep {} "{}"'.format(
                        self.target.busybox, report_end_id, ls_file
                    )
                    output = self.target.execute(
                        grep_cmd, as_root=True, check_exit_code=False
                    )
                    if report_end_id in output:
                        benchmark_complete = True
                        break

            iterations += 1

            if iterations > ((timeout_period_m * 60) // sleep_period_s):
                # We've been waiting 15 minutes for Speedometer to finish running - give up.
                if not local_storage_seen:
                    raise WorkloadError(
                        "Speedometer did not complete within 15m - Local Storage wasn't found"
                    )
                raise WorkloadError("Speedometer did not complete within 15 minutes.")

            time.sleep(sleep_period_s)

    def read_score(self):
        self.target.execute(
            "uiautomator dump {}".format(self.ui_dump_loc), as_root=True
        )
        self.target.pull(self.ui_dump_loc, self.temp_dir.name)

        with open(os.path.join(self.temp_dir.name, "ui_dump.xml"), "rb") as fh:
            dump = fh.read().decode("utf-8")
        match = self.regex.search(dump)
        result = None
        if match:
            result = float(match.group("value"))

        return result

    def update_output(self, context):
        super(Speedometer, self).update_output(context)

        self.ui_dump_loc = os.path.join(self.target.working_directory, "ui_dump.xml")

        score_read = False
        iterations = 0
        while not score_read:
            score = self.read_score()

            if score is not None:
                context.add_metric(
                    "Speedometer Score", score, "Runs per minute", lower_is_better=False
                )
                score_read = True
            else:
                if iterations >= 10:
                    raise WorkloadError(
                        "The Speedometer workload has failed. No score was obtainable."
                    )
                else:
                    # Sleep and retry...
                    time.sleep(2)
                    iterations += 1

    def teardown(self, context):
        super(Speedometer, self).teardown(context)

        # The browser's processes can stick around and have minor impact on
        # other performance sensitive workloads, so make sure we clean up.
        self.target.execute("am force-stop {}".format(self.chrome_package))

        if self.cleanup_assets:
            if self.ui_dump_loc is not None and self.target_file_was_created(
                self.ui_dump_loc
            ):
                # The only thing left on device was the UI dump created by uiautomator.
                self.target.execute("rm {}".format(self.ui_dump_loc), as_root=True)

        # Clear the cache we used to check if the local storage directory exists.
        self.target_file_was_seen.clear()
        self.ui_dump_loc = None

    @once
    def finalize(self, context):
        super(Speedometer, self).finalize(context)

        # Shutdown the locally hosted version of Speedometer
        self.archive_server.stop()


class ArchiveServerThread(threading.Thread):
    """Thread for running the HTTPServer"""

    def __init__(self, httpd):
        self._httpd = httpd
        threading.Thread.__init__(self)

    def run(self):
        self._httpd.serve_forever()


class DifferentDirectoryHTTPRequestHandler(SimpleHTTPRequestHandler):
    """A version of SimpleHTTPRequestHandler that allows us to serve
    relative files from a different directory than the current one.
    This directory is captured in |document_root|. It also suppresses
    logging."""

    def translate_path(self, path):
        document_root = self.server.document_root
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        requested_uri = os.path.relpath(path, os.getcwd())
        return os.path.join(document_root, requested_uri)

    # Disable the logging.
    # pylint: disable=redefined-builtin
    def log_message(self, format, *args):
        pass


class ArchiveServer(object):
    def __init__(self):
        self._port = None

    def start(self, document_root):
        # Create the server, and find out the port we've been assigned...
        self._httpd = HTTPServer(("", 0), DifferentDirectoryHTTPRequestHandler)
        # (This property is expected to be read by the
        #  DifferentDirectoryHTTPRequestHandler.translate_path method.)
        self._httpd.document_root = document_root
        _, self._port = self._httpd.server_address

        self._thread = ArchiveServerThread(self._httpd)
        self._thread.start()

    def stop(self):
        self._httpd.shutdown()
        self._thread.join()

    def expose_to_device(self, target):
        adb_command(target.adb_name, "reverse tcp:{0} tcp:{0}".format(self._port))

    def hide_from_device(self, target):
        adb_command(target.adb_name, "reverse --remove tcp:{}".format(self._port))

    def get_port(self):
        return self._port
