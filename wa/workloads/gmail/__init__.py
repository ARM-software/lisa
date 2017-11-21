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
from wa.framework.exception import ValidationError, WorkloadError


class Gmail(ApkUiautoWorkload):

    name = 'gmail'
    package_names = ['com.google.android.gm']
    description = '''
    A workload to perform standard productivity tasks within Gmail.  The workload carries out
    various tasks, such as creating new emails, attaching images and sending them.

    Test description:
    1. Open Gmail application
    2. Click to create New mail
    3. Attach an image from the local images folder to the email
    4. Enter recipient details in the To field
    5. Enter text in the Subject field
    6. Enter text in the Compose field
    7. Click the Send mail button

    Known working APK version: 7.11.5.176133587
    '''

    parameters = [
        Parameter('recipient', kind=str, default='wa-devnull@mailinator.com',
                  description='''
                  The email address of the recipient.  Setting a void address
                  will stop any mesage failures clogging up your device inbox
                  '''),
        Parameter('test_image', kind=str, default='uxperf_1600x1200.jpg',
                  description='''
                  An image to be copied onto the device that will be attached
                  to the email
                  '''),
    ]

    # This workload relies on the internet so check that there is a working
    # internet connection
    requires_network = True

    def __init__(self, target, **kwargs):
        super(Gmail, self).__init__(target, **kwargs)
        self.deployable_assets = [self.test_image]
        self.clean_assets = True

    def init_resources(self, context):
        super(Gmail, self).init_resources(context)
        if self.target.get_sdk_version() >= 24 and 'com.google.android.apps.photos' not in self.target.list_packages():
            raise WorkloadError('gmail workload requires Google Photos to be installed for Android N onwards')
        self.gui.uiauto_params['recipient'] = self.recipient
        # Only accept certain image formats
        if os.path.splitext(self.test_image.lower())[1] not in ['.jpg', '.jpeg', '.png']:
            raise ValidationError('{} must be a JPEG or PNG file'.format(self.test_image))
