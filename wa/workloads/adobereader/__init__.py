#    Copyright 2014-2016 ARM Limited
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

from wa import ApkUiautoWorkload, Parameter
from wa.utils.types import list_of_strs
from wa.framework.exception import ValidationError


class AdobeReader(ApkUiautoWorkload):

    name = 'adobereader'
    package_names = ['com.adobe.reader']
    description = '''
    The Adobe Reader workflow carries out the following typical productivity tasks.

    Test description:

    1. Open a local file on the device
    2. Gestures test:
        2.1. Swipe down across the central 50% of the screen in 200 x 5ms steps
        2.2. Swipe up across the central 50% of the screen in 200 x 5ms steps
        2.3. Swipe right from the edge of the screen in 50 x 5ms steps
        2.4. Swipe left from the edge of the screen  in 50 x 5ms steps
        2.5. Pinch out 50% in 100 x 5ms steps
        2.6. Pinch In 50% in 100 x 5ms steps
    3. Search test:
        Search ``document_name`` for each string in the ``search_string_list``
    4. Close the document

    Known working APK version: 16.1
    '''

    default_search_strings = [
        'The quick brown fox jumps over the lazy dog',
        'TEST_SEARCH_STRING',
    ]

    parameters = [
        Parameter('document_name', kind=str, default='uxperf_test_doc.pdf',
                  description='''
                  The document name to use for the Gesture and Search test.
                  '''),
        Parameter('search_string_list', kind=list_of_strs, default=default_search_strings,
                  constraint=lambda x: len(x) > 0,
                  description='''
                  For each string in the list, a document search is performed
                  using the string as the search term. At least one must be
                  provided.
                  '''),
    ]

    def __init__(self, target, **kwargs):
        super(AdobeReader, self).__init__(target, **kwargs)
        self.deployable_assets = [self.document_name]
        self.asset_directory = self.target.path.join(self.target.external_storage,
                                                     'Android', 'data',
                                                     'com.adobe.reader', 'files')
    def init_resources(self, context):
        super(AdobeReader, self).init_resources(context)
        # Only accept certain file formats
        if os.path.splitext(self.document_name.lower())[1] not in ['.pdf']:
            raise ValidationError('{} must be a PDF file'.format(self.document_name))
        self.gui.uiauto_params['filename'] = self.document_name
        self.gui.uiauto_params['search_string_list'] = self.search_string_list

    def setup(self, context):
        super(AdobeReader, self).setup(context)
        # Need to re-deploy each time to adobe folder as it is wiped upon clearing app
        self.deploy_assets(context)
