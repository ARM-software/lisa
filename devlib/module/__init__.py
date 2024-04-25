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

from devlib.exception import TargetStableError
from devlib.utils.types import identifier
from devlib.utils.misc import walk_modules

_module_registry = {}

def register_module(mod):
    if not issubclass(mod, Module):
        raise ValueError('A module must subclass devlib.Module')

    if mod.name is None:
        raise ValueError('A module must define a name')

    try:
        existing = _module_registry[mod.name]
    except KeyError:
        pass
    else:
        if existing is not mod:
            raise ValueError(f'Module "{mod.name}" already exists')
    _module_registry[mod.name] = mod


class Module:

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
        attr_name = cls.attr_name
        installed = target._installed_modules

        try:
            mod = installed[attr_name]
        except KeyError:
            mod = cls(target, **params)
            mod.logger.debug(f'Installing module {cls.name}')

            if mod.probe(target):
                for name in (
                    attr_name,
                    identifier(cls.name),
                    identifier(cls.kind) if cls.kind else None,
                ):
                    if name is not None:
                        installed[name] = mod

                target._modules[cls.name] = params
                return mod
            else:
                raise TargetStableError(f'Module "{cls.name}" is not supported by the target')
        else:
            raise ValueError(
                f'Attempting to install module "{cls.name}" but a module is already installed as attribute "{attr_name}": {mod}'
            )

    def __init__(self, target):
        self.target = target
        self.logger = logging.getLogger(self.name)


    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        attr_name = cls.kind or cls.name
        cls.attr_name = identifier(attr_name) if attr_name else None

        if cls.name is not None:
            register_module(cls)


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
    def from_registry(mod):
        try:
            return _module_registry[mod]
        except KeyError:
            raise ValueError('Module "{}" does not exist'.format(mod))

    if isinstance(mod, str):
        try:
            return from_registry(mod)
        except ValueError:
            # If the lookup failed, we may have simply not imported Modules
            # from the devlib.module package. The former module loading
            # implementation was also pre-importing modules, so we need to
            # replicate that behavior since users are currently not expected to
            # have imported the module prior to trying to use it.
            walk_modules('devlib.module')
            return from_registry(mod)

    elif issubclass(mod, Module):
        return mod
    else:
        raise ValueError('Not a valid module: {}'.format(mod))
