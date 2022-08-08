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

from wa import Parameter, ApkUiautoWorkload
from wa.framework.exception import WorkloadError


class Chrome(ApkUiautoWorkload):

    name = 'chrome'
    description = '''
    A workload to perform standard Web browsing tasks with Google Chrome. The
    workload carries out a number of typical Web-based tasks, navigating through
    a handful of Wikipedia pages in multiple browser tabs.

    To run the workload in offline mode, a ``pages.tar`` archive and an
    ``OfflinePages.db`` file are required. For users wishing to generate these
    files themselves, Chrome should first be operated from an Internet-connected
    environment and the following Wikipedia pages should be downloaded for
    offline use within Chrome:

    - https://en.m.wikipedia.org/wiki/Main_Page
    - https://en.m.wikipedia.org/wiki/United_States
    - https://en.m.wikipedia.org/wiki/California

    Following this, the files of interest for viewing these pages offline can be
    found in the ``/data/data/com.android.chrome/app_chrome/Default/Offline
    Pages`` directory. The ``OfflinePages.db`` file can be copied from the
    'metadata' subdirectory, while the ``*.mhtml`` files that should make up the
    ``pages.tar`` file can be found in the 'archives' subdirectory. These page
    files can then be archived to produce a tarball using a command such as
    ``tar -cvf pages.tar -C /path/to/archives .``.  Both this and
    ``OfflinePages.db`` should then be placed in the
    ``~/.workload_automation/dependencies/chrome/`` directory on your local
    machine, creating this if it does not already exist.

    Known working APK version: 65.0.3325.109
    '''
    package_names = ['com.android.chrome']

    parameters = [
        Parameter('offline_mode', kind=bool, default=False, description='''
                  If set to ``True``, the workload will execute in offline mode.
                  This mode requires root and makes use of a tarball of \*.mhtml
                  files 'pages.tar' and an metadata database 'OfflinePages.db'.
                  The tarball is extracted directly to the application's offline
                  pages 'archives' directory, while the database is copied to
                  the offline pages 'metadata' directory.
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
        super(Chrome, self).__init__(target, **kwargs)
        if self.offline_mode:
            self.deployable_assets = ['pages.tar', 'OfflinePages.db']
            self.cleanup_assets = True

    def initialize(self, context):
        super(Chrome, self).initialize(context)
        if self.offline_mode and not self.target.is_rooted:
            raise WorkloadError('This workload requires root to set up Chrome for offline usage.')

    def setup_rerun(self):
        super(Chrome, self).setup_rerun()
        offline_pages = self.target.path.join(self.target.package_data_directory, self.package, 'app_chrome', 'Default', 'Offline\ Pages')
        metadata_src = self.target.path.join(self.target.working_directory, 'OfflinePages.db')
        metadata_dst = self.target.path.join(offline_pages, 'metadata')
        archives_src = self.target.path.join(self.target.working_directory, 'pages.tar')
        archives_dst = self.target.path.join(offline_pages, 'archives')
        owner = self.target.execute("{} stat -c '%u' {}".format(self.target.busybox, offline_pages), as_root=True).strip()
        self.target.execute('{} tar -xvf {} -C {}'.format(self.target.busybox, archives_src, archives_dst), as_root=True)
        self.target.execute('{} cp {} {}'.format(self.target.busybox, metadata_src, metadata_dst), as_root=True)
        self.target.execute('{0} chown -R {1}:{1} {2}'.format(self.target.busybox, owner, offline_pages), as_root=True)
