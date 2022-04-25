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
import logging
from inspect import isclass

from past.builtins import basestring

from devlib.utils.misc import walk_modules
from devlib.utils.types import identifier


__module_cache = {}


class Module(object):

    name = None
    kind = None
    # This is the stage at which the module will be installed. Current valid
    # stages are:
    #  'early' -- installed when the Target is first created. This should be
    #             used for modules that do not rely on the main connection
    #             being established (usually because the commumnitcate with the
    #             target through some sorto of secondary connection, e.g. via
    #             serial).
    #  'connected' -- installed when a connection to to the target has been
    #                 established. This is the default.
    #   'setup' -- installed after initial setup of the device has been performed.
    #              This allows the module to utilize assets deployed during the
    #              setup stage for example 'Busybox'.
    stage = 'connected'

    @staticmethod
    def probe(target):
        raise NotImplementedError()

    @classmethod
    def install(cls, target, **params):
        if cls.kind is not None:
            attr_name = identifier(cls.kind)
        else:
            attr_name = identifier(cls.name)
        if hasattr(target, attr_name):
            existing_module = getattr(target, attr_name)
            existing_name = getattr(existing_module, 'name', str(existing_module))
            message = 'Attempting to install module "{}" which already exists (new: {}, existing: {})'
            raise ValueError(message.format(attr_name, cls.name, existing_name))
        setattr(target, attr_name, cls(target, **params))

    def __init__(self, target):
        self.target = target
        self.logger = logging.getLogger(self.name)


class HardRestModule(Module):

    kind = 'hard_reset'

    def __call__(self):
        raise NotImplementedError()


class BootModule(Module):

    kind = 'boot'

    def __call__(self):
        raise NotImplementedError()

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise ValueError('Unknown parameter "{}" for {}'.format(name, self.name))
            self.logger.debug('Updating "{}" to "{}"'.format(name, value))
            setattr(self, name, value)


class FlashModule(Module):

    kind = 'flash'

    def __call__(self, image_bundle=None, images=None, boot_config=None, connect=True):
        raise NotImplementedError()


def get_module(mod):
    if not __module_cache:
        __load_cache()

    if isinstance(mod, basestring):
        try:
            return __module_cache[mod]
        except KeyError:
            raise ValueError('Module "{}" does not exist'.format(mod))
    elif issubclass(mod, Module):
        return mod
    else:
        raise ValueError('Not a valid module: {}'.format(mod))


def register_module(mod):
    if not issubclass(mod, Module):
        raise ValueError('A module must subclass devlib.Module')
    if mod.name is None:
        raise ValueError('A module must define a name')
    if mod.name in __module_cache:
        raise ValueError('Module {} already exists'.format(mod.name))
    __module_cache[mod.name] = mod


def __load_cache():
    for module in walk_modules('devlib.module'):
        for obj in vars(module).values():
            if isclass(obj) and issubclass(obj, Module) and obj.name:
                register_module(obj)
