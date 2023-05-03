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

    To run the workload in offline mode, a 'mailstore.tar' file is required. In order to
    generate such a file, Gmail should first be operated from an Internet-connected environment.
    After this, the relevant database files can be found in the
    '/data/data/com.google.android.gm/databases' directory. These files can then be archived to
    produce a tarball using a command such as ``tar -cvf mailstore.tar -C /path/to/databases .``.
    The result should then be placed in the '~/.workload_automation/dependencies/gmail/' directory
    on your local machine, creating this if it does not already exist.

    Known working APK version: 2023.04.02.523594694.Release
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
        Parameter('offline_mode', kind=bool, default=False, description='''
                  If set to ``True``, the workload will execute in offline mode.
                  This mode requires root and makes use of a tarball of email
                  database files 'mailstore.tar' for the email account to be used.
                  This file is extracted directly to the application's 'databases'
                  directory at '/data/data/com.google.android.gm/databases'.
                  '''),
    ]

    @property
    def requires_network(self):
        return not self.offline_mode

    @property
    def requires_rerun(self):
        # In offline mode we need to restart the application after modifying its data directory
        return self.offline_mode

    def __init__(self, target, **kwargs):
        super(Gmail, self).__init__(target, **kwargs)
        self.deployable_assets = [self.test_image]
        if self.offline_mode:
            self.deployable_assets.append('mailstore.tar')
        self.cleanup_assets = True

    def initialize(self, context):
        super(Gmail, self).initialize(context)
        if self.offline_mode and not self.target.is_rooted:
            raise WorkloadError('This workload requires root to set up Gmail for offline usage.')

    def init_resources(self, context):
        super(Gmail, self).init_resources(context)
        # Allows for getting working directory regardless if path ends with a '/'
        work_dir = self.target.working_directory
        work_dir = work_dir if work_dir[-1] != os.sep else work_dir[:-1]
        self.gui.uiauto_params['workdir_name'] = self.target.path.basename(work_dir)
        self.gui.uiauto_params['recipient'] = self.recipient
        self.gui.uiauto_params['offline_mode'] = self.offline_mode
        self.gui.uiauto_params['test_image'] = self.test_image
        # Only accept certain image formats
        if os.path.splitext(self.test_image.lower())[1] not in ['.jpg', '.jpeg', '.png']:
            raise ValidationError('{} must be a JPEG or PNG file'.format(self.test_image))

    def setup_rerun(self):
        super(Gmail, self).setup_rerun()
        database_src = self.target.path.join(self.target.working_directory, 'mailstore.tar')
        database_dst = self.target.path.join(self.target.package_data_directory, self.package, 'databases')
        existing_mailstores = self.target.path.join(database_dst, 'mailstore.*')
        owner = self.target.execute("{} stat -c '%u' {}".format(self.target.busybox, database_dst), as_root=True).strip()
        self.target.execute('{} rm {}'.format(self.target.busybox, existing_mailstores), as_root=True)
        self.target.execute('{} tar -xvf {} -C {}'.format(self.target.busybox, database_src, database_dst), as_root=True)
        self.target.execute('{0} chown -R {1}:{1} {2}'.format(self.target.busybox, owner, database_dst), as_root=True)
