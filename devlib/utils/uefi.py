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


import re
import time
import logging
from copy import copy

from past.builtins import basestring

from devlib.utils.serial_port import write_characters, TIMEOUT
from devlib.utils.types import boolean


logger = logging.getLogger('UEFI')


class UefiConfig(object):

    def __init__(self, config_dict):
        if isinstance(config_dict, UefiConfig):
            self.__dict__ = copy(config_dict.__dict__)
        else:
            try:
                self.image_name = config_dict['image_name']
                self.image_args = config_dict['image_args']
                self.fdt_support = boolean(config_dict['fdt_support'])
            except KeyError as e:
                raise ValueError('Missing mandatory parameter for UEFI entry config: "{}"'.format(e))
            self.initrd = config_dict.get('initrd')
            self.fdt_path = config_dict.get('fdt_path')
            if self.fdt_path and not self.fdt_support:
                raise ValueError('FDT path has been specfied for UEFI entry, when FDT support is "False"')


class UefiMenu(object):
    """
    Allows navigating UEFI menu over serial (it relies on a pexpect connection).

    """

    option_regex = re.compile(r'^\[(\d+)\]\s+([^\r]+)\r\n', re.M)
    prompt_regex = re.compile(r'^(\S[^\r\n]+):\s*', re.M)
    invalid_regex = re.compile(r'Invalid input \(max (\d+)\)', re.M)

    load_delay = 1  # seconds
    default_timeout = 60  # seconds

    def __init__(self, conn, prompt='The default boot selection will start in'):
        """
        :param conn: A serial connection as returned by ``pexect.spawn()``.
        :param prompt: The starting prompt to wait for during ``open()``.

        """
        self.conn = conn
        self.start_prompt = prompt
        self.options = {}
        self.prompt = None
        self.attempting_invalid_retry = False

    def wait(self, timeout=default_timeout):
        """
        "Open" the UEFI menu by sending an interrupt on STDIN after seeing the
        starting prompt (configurable upon creation of the ``UefiMenu`` object.

        """
        self.conn.expect(self.start_prompt, timeout)
        self.connect()

    def connect(self, timeout=default_timeout):
        self.nudge()
        time.sleep(self.load_delay)
        self.read_menu(timeout=timeout)

    def create_entry(self, name, config):
        """Create a new UEFI entry using the parameters. The menu is assumed
        to be at the top level. Upon return, the menu will be at the top level."""
        logger.debug('Creating UEFI entry {}'.format(name))
        self.nudge()
        self.select('Boot Manager')
        self.select('Add Boot Device Entry')
        self.select('NOR Flash')
        self.enter(config.image_name)
        self.enter('y' if config.fdt_support else 'n')
        if config.initrd:
            self.enter('y')
            self.enter(config.initrd)
        else:
            self.enter('n')
        self.enter(config.image_args)
        self.enter(name)

        if config.fdt_path:
            self.select('Update FDT path')
            self.enter(config.fdt_path)

        self.select('Return to main menu')

    def delete_entry(self, name):
        """Delete the specified UEFI entry. The menu is assumed
        to be at the top level. Upon return, the menu will be at the top level."""
        logger.debug('Removing UEFI entry {}'.format(name))
        self.nudge()
        self.select('Boot Manager')
        self.select('Remove Boot Device Entry')
        self.select(name)
        self.select('Return to main menu')

    def select(self, option, timeout=default_timeout):
        """
        Select the specified option from the current menu.

        :param option: Could be an ``int`` index of the option, or a string/regex to
                       match option text against.
        :param timeout: If a non-``int`` option is specified, the option list may need
                        need to be parsed (if it hasn't been already), this may block
                        and the timeout is used to cap that , resulting in a ``TIMEOUT``
                        exception.
        :param delay: A fixed delay to wait after sending the input to the serial connection.
                      This should be set if input this action is known to result in a
                      long-running operation.

        """
        if isinstance(option, basestring):
            option = self.get_option_index(option, timeout)
        self.enter(option)

    def enter(self, value, delay=load_delay):
        """Like ``select()`` except no resolution is performed -- the value is sent directly
        to the serial connection."""
        # Empty the buffer first, so that only response to the input about to
        # be sent will be processed by subsequent commands.
        value = str(value)
        self._reset()
        write_characters(self.conn, value)
        # TODO: in case the value is long an complicated, things may get
        # screwed up (e.g. there may be line breaks injected), additionally,
        # special chars might cause regex to fail. To avoid these issues i'm
        # only matching against the first 5 chars of the value. This is
        # entirely arbitrary and I'll probably have to find a better way of
        # doing this at some point.
        self.conn.expect(value[:5], timeout=delay)
        time.sleep(self.load_delay)

    def read_menu(self, timeout=default_timeout):
        """Parse serial output to get the menu options and the following prompt."""
        attempting_timeout_retry = False
        self.attempting_invalid_retry = False
        while True:
            index = self.conn.expect([self.option_regex, self.prompt_regex, self.invalid_regex, TIMEOUT],
                                     timeout=timeout)
            match = self.conn.match
            if index == 0:  # matched menu option
                self.options[match.group(1)] = match.group(2)
            elif index == 1:  # matched prompt
                self.prompt = match.group(1)
                break
            elif index == 2:  # matched invalid selection
                # We've sent an invalid input (which includes an empty line) at
                # the top-level menu. To get back the menu options, it seems we
                # need to enter what the error reports as the max + 1, so...
                if not self.attempting_invalid_retry:
                    self.attempting_invalid_retry = True
                    val = int(match.group(1))
                    self.empty_buffer()
                    self.enter(val)
                    self.select('Return to main menu')
                    self.attempting_invalid_retry = False
                else:   # OK, that didn't work; panic!
                    raise RuntimeError('Could not read menu entries stuck on "{}" prompt'.format(self.prompt))
            elif index == 3:  # timed out
                if not attempting_timeout_retry:
                    attempting_timeout_retry = True
                    self.nudge()
                else:  # Didn't help. Run away!
                    raise RuntimeError('Did not see a valid UEFI menu.')
            else:
                raise AssertionError('Unexpected response waiting for UEFI menu')  # should never get here

    def list_options(self, timeout=default_timeout):
        """Returns the menu index of the specified option text (uses regex matching). If the option
        is not in the current menu, ``LookupError`` will be raised."""
        if not self.prompt:
            self.read_menu(timeout)
        return list(self.options.items())

    def get_option_index(self, text, timeout=default_timeout):
        """Returns the menu index of the specified option text (uses regex matching). If the option
        is not in the current menu, ``LookupError`` will be raised."""
        if not self.prompt:
            self.read_menu(timeout)
        for k, v in self.options.items():
            if re.search(text, v):
                return k
        raise LookupError(text)

    def has_option(self, text, timeout=default_timeout):
        """Returns ``True`` if at least one of the options in the current menu has
        matched (using regex) the specified text."""
        try:
            self.get_option_index(text, timeout)
            return True
        except LookupError:
            return False

    def nudge(self):
        """Send a little nudge to ensure there is something to read. This is useful when you're not
        sure if all out put from the serial has been read already."""
        self.enter('')

    def empty_buffer(self):
        """Read everything from the serial and clear the internal pexpect buffer. This ensures
        that the next ``expect()`` call will time out (unless further input will be sent to the
        serial beforehand. This is used to create a "known" state and avoid unexpected matches."""
        try:
            while True:
                time.sleep(0.1)
                self.conn.read_nonblocking(size=1024, timeout=0.1)
        except TIMEOUT:
            pass
        self.conn.buffer = ''

    def _reset(self):
        self.options = {}
        self.prompt = None
        self.empty_buffer()
