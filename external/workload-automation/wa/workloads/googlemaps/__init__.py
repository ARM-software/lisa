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


class GoogleMaps(ApkUiautoWorkload):

    name = 'googlemaps'
    description = '''
    A workload to perform standard navigation tasks with Google Maps. This workload searches
    for known locations, pans and zooms around the map, and follows driving directions
    along a route.

    To run the workload in offline mode, ``databases.tar`` and ``files.tar`` archives are required.
    In order to generate these files, Google Maps should first be operated from an
    Internet-connected environment, and a region around Cambridge, England should be downloaded
    for offline use. This region must include the landmarks used in the UIAutomator program,
    which include Cambridge train station and Corpus Christi college.

    Following this, the files of interest can be found in the ``databases`` and ``files``
    subdirectories of the ``/data/data/com.google.android.apps.maps/`` directory. The contents
    of these subdirectories can be archived into tarballs using commands such as
    ``tar -cvf databases.tar -C /path/to/databases .``. These ``databases.tar`` and ``files.tar`` archives
    should then be placed in the ``~/.workload_automation/dependencies/googlemaps`` directory on your
    local machine, creating this if it does not already exist.

    Known working APK version: 10.19.1
    '''
    package_names = ['com.google.android.apps.maps']

    parameters = [
        Parameter('offline_mode', kind=bool, default=False, description='''
                  If set to ``True``, the workload will execute in offline mode.
                  This mode requires root and makes use of a tarball of database
                  files ``databases.tar`` and a tarball of auxiliary files ``files.tar``.
                  These tarballs are extracted directly to the application's ``databases``
                  and ``files`` directories respectively in ``/data/data/com.google.android.apps.maps/``.
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
        super(GoogleMaps, self).__init__(target, **kwargs)
        if self.offline_mode:
            self.deployable_assets = ['databases.tar', 'files.tar']
            self.cleanup_assets = True

    def initialize(self, context):
        super(GoogleMaps, self).initialize(context)
        if self.offline_mode and not self.target.is_rooted:
            raise WorkloadError('This workload requires root to set up Google Maps for offline usage.')

    def init_resources(self, context):
        super(GoogleMaps, self).init_resources(context)
        self.gui.uiauto_params['offline_mode'] = self.offline_mode

    def setup_rerun(self):
        super(GoogleMaps, self).setup_rerun()
        package_data_dir = self.target.path.join(self.target.package_data_directory, self.package)
        databases_src = self.target.path.join(self.target.working_directory, 'databases.tar')
        databases_dst = self.target.path.join(package_data_dir, 'databases')
        files_src = self.target.path.join(self.target.working_directory, 'files.tar')
        files_dst = self.target.path.join(package_data_dir, 'files')
        owner = self.target.execute("{} stat -c '%u' {}".format(self.target.busybox, package_data_dir), as_root=True).strip()
        self.target.execute('{} tar -xvf {} -C {}'.format(self.target.busybox, databases_src, databases_dst), as_root=True)
        self.target.execute('{} tar -xvf {} -C {}'.format(self.target.busybox, files_src, files_dst), as_root=True)
        self.target.execute('{0} chown -R {1}:{1} {2}'.format(self.target.busybox, owner, package_data_dir), as_root=True)
