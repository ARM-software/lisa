#    Copyright 2014-2017 ARM Limited
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


class GoogleSlides(ApkUiautoWorkload):

    name = 'googleslides'
    package_names = ['com.google.android.apps.docs.editors.slides']

    description = '''
    A workload to perform standard productivity tasks with Google Slides. The workload carries
    out various tasks, such as creating a new presentation, adding text, images, and shapes,
    as well as basic editing and playing a slideshow.
    This workload should be able to run without a network connection.

    There are two main scenarios:
      1. create test: a presentation is created in-app and some editing done on it,
      2. load test: a pre-existing PowerPoint file is copied onto the device for testing.

    --- create ---
    Create a new file in the application and perform basic editing on it. This test also
    requires an image file specified by the param ``test_image`` to be copied onto the device.

    Test description:

    1. Start the app and skip the welcome screen. Dismiss the work offline banner if present.
    2. Go to the app settings page and enables PowerPoint compatibility mode. This allows
       PowerPoint files to be created inside Google Slides.
    3. Create a new PowerPoint presentation in the app (PPT compatibility mode) with a title
       slide and save it to device storage.
    4. Insert another slide and to it insert the pushed image by picking it from the gallery.
    5. Insert a final slide and add a shape to it. Resize and drag the shape to modify it.
    6. Finally, navigate back to the documents list.

    --- load ---
    Copy a PowerPoint presentation onto the device to test slide navigation. The PowerPoint
    file to be copied is given by ``test_file``.

    Test description:

    1. From the documents list (following the create test), open the specified PowerPoint
       by navigating into device storage and wait for it to be loaded.
    2. A navigation test is performed while the file is in editing mode (i.e. not slideshow).
       swiping forward to the next slide until ``slide_count`` swipes are performed.
    3. While still in editing mode, the same action is done in the reverse direction back to
       the first slide.
    4. Enter presentation mode by selecting to play the slideshow.
    5. Swipe forward to play the slideshow, for a maximum number of ``slide_count`` swipes.
    6. Finally, repeat the previous step in the reverse direction while still in presentation
       mode, navigating back to the first slide.

    NOTE: There are known issues with the reliability of this workload on some targets.
    It MAY NOT ALWAYS WORK on your device. If you do run into problems, it might help to
    set ``do_text_entry`` parameter to ``False``.

    Known working APK version: 1.20.442.04.40
    '''

    parameters = [
        Parameter('test_image', kind=str, default='uxperf_1600x1200.jpg',
                  description='''
                  An image to be copied onto the device that will be embedded in the
                  PowerPoint file as part of the test.
                  '''),
        Parameter('test_file', kind=str, default='uxperf_test_doc.pptx',
                  description='''
                  If specified, the workload will copy the PowerPoint file to be used for
                  testing onto the device. Otherwise, a file will be created inside the app.
                  '''),
        Parameter('slide_count', kind=int, default=5,
                  description='''
                  Number of slides in aforementioned local file. Determines number of
                  swipe actions when playing slide show.
                  '''),
        Parameter('do_text_entry', kind=bool, default=True,
                  description='''
                  If set to ``True``, will attempt to enter text in the first slide as part
                  of the test. Currently seems to be problematic on some devices, most
                  notably Samsung devices.
                  ''')
    ]

    # Created file will be saved with this name
    new_doc_name = "WORKLOAD AUTOMATION"

    def __init__(self, target, **kwargs):
        super(GoogleSlides, self).__init__(target, **kwargs)
        self.run_timeout = 600
        self.deployable_assets = [self.test_image, self.test_file]

    def init_resources(self, context):
        super(GoogleSlides, self).init_resources(context)
        # Allows for getting working directory regardless if path ends with a '/'
        work_dir = self.target.working_directory
        work_dir = work_dir if work_dir[-1] != os.sep else work_dir[:-1]
        self.gui.uiauto_params['workdir_name'] = self.target.path.basename(work_dir)
        self.gui.uiauto_params['test_file'] = self.test_file
        self.gui.uiauto_params['slide_count'] = self.slide_count
        self.gui.uiauto_params['do_text_entry'] = self.do_text_entry
        self.gui.uiauto_params['new_doc_name'] = self.new_doc_name
        # Only accept certain image formats
        if os.path.splitext(self.test_image.lower())[1] not in ['.jpg', '.jpeg', '.png']:
            raise ValidationError('{} must be a JPEG or PNG file'.format(self.test_image))
        # Only accept certain presentation formats
        if os.path.splitext(self.test_file.lower())[1] not in ['.pptx']:
            raise ValidationError('{} must be a PPTX file'.format(self.test_file))
