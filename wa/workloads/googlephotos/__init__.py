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
from wa.framework.exception import ValidationError
from wa.utils.types import list_of_strs
from wa.utils.misc import unique


class Googlephotos(ApkUiautoWorkload):

    name = 'googlephotos'
    package_names = ['com.google.android.apps.photos']
    description = '''
    A workload to perform standard productivity tasks with Google Photos. The workload carries out
    various tasks, such as browsing images, performing zooms, and post-processing the image.

    Test description:

    1. Four images are copied to the target
    2. The application is started in offline access mode
    3. Gestures are performed to pinch zoom in and out of the selected image
    4. The colour of a selected image is edited by selecting the colour menu, incrementing the
       colour, resetting the colour and decrementing the colour using the seek bar.
    5. A crop test is performed on a selected image.  UiAutomator does not allow the selection of
       the crop markers so the image is tilted positively, reset and then tilted negatively to get a
       similar cropping effect.
    6. A rotate test is performed on a selected image, rotating anticlockwise 90 degrees, 180
       degrees and 270 degrees.

    Known working APK version: 1.21.0.123444480
    '''

    default_test_images = [
        'uxperf_1200x1600.png', 'uxperf_1600x1200.jpg',
        'uxperf_2448x3264.png', 'uxperf_3264x2448.jpg',
    ]

    parameters = [
        Parameter('test_images', kind=list_of_strs, default=default_test_images,
                  constraint=lambda x: len(unique(x)) == 4,
                  description='''
                  A list of four JPEG and/or PNG files to be pushed to the target.
                  Absolute file paths may be used but tilde expansion must be escaped.
                  '''),
    ]

    def __init__(self, target, **kwargs):
        super(Googlephotos, self).__init__(target, **kwargs)
        self.deployable_assets = self.test_images

    def init_resources(self, context):
        super(Googlephotos, self).init_resources(context)
        # Only accept certain image formats
        for image in self.test_images:
            if os.path.splitext(image.lower())[1] not in ['.jpg', '.jpeg', '.png']:
                raise ValidationError('{} must be a JPEG or PNG file'.format(image))

    def deploy_assets(self, context):
        super(Googlephotos, self).deploy_assets(context)
        # Create a subfolder for each test_image named ``wa-[1-4]``
        # Move each image into its subfolder
        # This is to guarantee ordering and allows the workload to select a specific
        # image by subfolder, as filenames are not shown easily within the app
        d = self.target.working_directory
        e = self.target.external_storage

        file_list = []

        for i, f in enumerate(self.test_images):
            orig_file_path = self.target.path.join(d, f)
            new_dir = self.target.path.join(e, 'wa', 'wa-{}'.format(i+1))
            new_file_path = self.target.path.join(new_dir, f)

            self.target.execute('mkdir -p {}'.format(new_dir))
            self.target.execute('cp {} {}'.format(orig_file_path, new_file_path))
            self.target.execute('rm {}'.format(orig_file_path))
            file_list.append(new_file_path)
        self.deployed_assets = file_list
        # Force rescan
        self.target.refresh_files(self.deployed_assets)

    def remove_assets(self, context):
        for asset in self.deployed_assets:
            self.target.remove(os.path.dirname(asset))
        self.target.refresh_files(self.deployed_assets)
