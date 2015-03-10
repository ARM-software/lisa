#    Copyright 2014-2015 ARM Limited
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

import time

from wlauto import UiAutomatorWorkload, Parameter
from wlauto.utils.types import boolean


class SkypeVideo(UiAutomatorWorkload):

    name = 'skypevideo'
    description = """
    Initiates Skype video call to a specified contact for a pre-determined duration.
    (Note: requires Skype to be set up appropriately).

    This workload is intended for monitoring the behaviour of a device while a Skype
    video call is in progress (a common use case). It does not produce any score or
    metric and the intention is that some addition instrumentation is enabled while
    running this workload.

    This workload, obviously, requires a network connection (ideally, wifi).

    This workload accepts the following parameters:


    **Skype Setup**

       - You should install Skype client from Google Play Store on the device
         (this was tested with client version 4.5.0.39600; other recent versions
         should also work).
       - You must have an account set up and logged into Skype on the device.
       - The contact to be called must be added (and has accepted) to the
         account. It's possible to have multiple contacts in the list, however
         the contact to be called *must* be visible on initial navigation to the
         list.
       - The contact must be able to received the call. This means that there
         must be  a Skype client running (somewhere) with the contact logged in
         and that client must have been configured to auto-accept calls from the
         account on the device (how to set this varies between different versions
         of Skype and between platforms -- please search online for specific
         instructions).
         https://support.skype.com/en/faq/FA3751/can-i-automatically-answer-all-my-calls-with-video-in-skype-for-windows-desktop

    """

    package = 'com.skype.raider'

    parameters = [
        Parameter('duration', kind=int, default=300,
                  description='Duration of the video call in seconds.'),
        Parameter('contact', mandatory=True,
                  description="""
                  The name of the Skype contact to call. The contact must be already
                  added (see below). *If use_gui is set*, then this must be the skype
                  ID of the contact, *otherwise*, this must be the name of the
                  contact as it appears in Skype client's contacts list. In the latter case
                  it *must not* contain underscore characters (``_``); it may, however, contain
                  spaces. There is no default, you **must specify the name of the contact**.

                  .. note:: You may alternatively specify the contact name as
                            ``skype_contact`` setting in your ``config.py``. If this is
                            specified, the ``contact`` parameter is optional, though
                            it may still be specified (in which case it will override
                            ``skype_contact`` setting).
                  """),
        Parameter('use_gui', kind=boolean, default=False,
                  description="""
                  Specifies whether the call should be placed directly through a
                  Skype URI, or by navigating the GUI. The URI is the recommended way
                  to place Skype calls on a device, but that does not seem to work
                  correctly on some devices (the URI seems to just start Skype, but not
                  place the call), so an alternative exists that will start the Skype app
                  and will then navigate the UI to place the call (incidentally, this method
                  does not seem to work on all devices either, as sometimes Skype starts
                  backgrounded...). Please note that the meaning of ``contact`` prameter
                  is different depending on whether this is set.  Defaults to ``False``.

                  .. note:: You may alternatively specify this as ``skype_use_gui`` setting
                            in your ``config.py``.
                  """),

    ]

    def __init__(self, device, **kwargs):
        super(SkypeVideo, self).__init__(device, **kwargs)
        if self.use_gui:
            self.uiauto_params['name'] = self.contact.replace(' ', '_')
            self.uiauto_params['duration'] = self.duration
        self.run_timeout = self.duration + 30

    def setup(self, context):
        if self.use_gui:
            super(SkypeVideo, self).setup(context)
            self.device.execute('am force-stop {}'.format(self.package))
            self.device.execute('am start -W -a android.intent.action.VIEW -d skype:')
        else:
            self.device.execute('am force-stop {}'.format(self.package))

    def run(self, context):
        if self.use_gui:
            super(SkypeVideo, self).run(context)
        else:
            command = "am start -W -a android.intent.action.VIEW -d \"skype:{}?call&video=true\""
            self.logger.debug(self.device.execute(command.format(self.contact)))
            self.logger.debug('Call started; waiting for {} seconds...'.format(self.duration))
            time.sleep(self.duration)
            self.device.execute('am force-stop com.skype.raider')

    def update_result(self, context):
        pass

    def teardown(self, context):
        if self.use_gui:
            super(SkypeVideo, self).teardown(context)
        self.device.execute('am force-stop {}'.format(self.package))
