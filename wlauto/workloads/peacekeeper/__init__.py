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

# pylint: disable=E1101,W0201,E0203
import os
import urllib2
from HTMLParser import HTMLParser

from wlauto import AndroidUiAutoBenchmark, Parameter
from wlauto.exceptions import WorkloadError


BROWSER_MAP = {
    'firefox': {
        'package': 'org.mozilla.firefox',
        'activity': '.App',
    },
    'chrome': {
        'package': 'com.android.chrome',
        'activity': 'com.google.android.apps.chrome.Main',
    },
}


class Peacekeeper(AndroidUiAutoBenchmark):

    name = 'peacekeeper'
    description = """
    Peacekeeper is a free and fast browser test that measures a browser's speed.

    .. note::

       This workload requires a network connection as well as support for
       one of the two currently-supported browsers. Moreover, TC2 has
       compatibility issue with chrome

    """
    run_timeout = 15 * 60

    parameters = [
        Parameter('browser', default='firefox', allowed_values=['firefox', 'chrome'],
                  description='The browser to be benchmarked.'),
        Parameter('output_file', default=None,
                  description="""The result URL of peacekeeper benchmark will be written
                                 into this file on device after completion of peacekeeper benchmark.
                                 Defaults to peacekeeper.txt in the device's ``working_directory``.
                  """),
        Parameter('peacekeeper_url', default='http://peacekeeper.futuremark.com/run.action',
                  description='The URL to run the peacekeeper benchmark.'),
    ]

    def __init__(self, device, **kwargs):
        super(Peacekeeper, self).__init__(device, **kwargs)
        self.version = self.browser

    def update_result(self, context):
        super(Peacekeeper, self).update_result(context)
        url = None

        # Pull the result page url, which contains the results, from the
        # peacekeeper.txt file and process it
        self.device.pull(self.output_file, context.output_directory)
        result_file = os.path.join(context.output_directory, 'peacekeeper.txt')
        with open(result_file) as fh:
            for line in fh:
                url = line

        # Fetch the html page containing the results
        if not url:
            raise WorkloadError('The url is empty, error while running peacekeeper benchmark')

        req = urllib2.Request(url)
        response = urllib2.urlopen(req)
        result_page = response.read()

        # Parse the HTML content using HTML parser
        parser = PeacekeeperParser()
        parser.feed(result_page)

        # Add peacekeeper_score into results file
        context.result.add_metric('peacekeeper_score', parser.peacekeeper_score)

    def validate(self):
        if self.output_file is None:
            self.output_file = os.path.join(self.device.working_directory, 'peacekeeper.txt')
        if self.browser == 'chrome' and self.device == 'TC2':
            raise WorkloadError('Chrome not supported on TC2')

        self.uiauto_params['output_file'] = self.output_file
        self.uiauto_params['browser'] = self.browser
        self.uiauto_params['peacekeeper_url'] = self.peacekeeper_url

        self.package = BROWSER_MAP[self.browser]['package']
        self.activity = BROWSER_MAP[self.browser]['activity']


class PeacekeeperParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.flag = False
        self.peacekeeper_score = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            for name, value in attrs:
                if name == 'class' and value == 'resultBarContainer clearfix resultBarSelected':
                    self.flag = True
                elif self.flag and name == 'class' and value == 'resultBarComment':
                    self.flag = False
                    self.peacekeeper_score = self.peacekeeper_score.split('details')[1]

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        if self.flag:
            self.peacekeeper_score += data.strip()
