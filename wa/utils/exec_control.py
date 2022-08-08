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

# "environment" management:
__environments = {}
__active_environment = None


def activate_environment(name):
    """
    Sets the current tracking environment to ``name``. If an
    environment with that name does not already exist, it will be
    created.
    """
    # pylint: disable=W0603
    global __active_environment

    if name not in list(__environments.keys()):
        init_environment(name)
    __active_environment = name


def init_environment(name):
    """
    Create a new environment called ``name``, but do not set it as the
    current environment.

    :raises: ``ValueError`` if an environment with name ``name``
             already exists.
    """
    if name in list(__environments.keys()):
        msg = "Environment {} already exists".format(name)
        raise ValueError(msg)
    __environments[name] = []


def reset_environment(name=None):
    """
    Reset method call tracking for environment ``name``. If ``name`` is
    not specified or is ``None``, reset the current active environment.

    :raises: ``ValueError`` if an environment with name ``name``
          does not exist.
    """

    if name is not None:
        if name not in list(__environments.keys()):
            msg = "Environment {} does not exist".format(name)
            raise ValueError(msg)
        __environments[name] = []
    else:
        if __active_environment is None:
            activate_environment('default')
        __environments[__active_environment] = []


# The decorators:
def once_per_instance(method):
    """
    The specified method will be invoked only once for every bound
    instance within the environment.
    """
    def wrapper(*args, **kwargs):
        if __active_environment is None:
            activate_environment('default')
        func_id = repr(method.__hash__()) + repr(args[0].__hash__())
        if func_id in __environments[__active_environment]:
            return
        else:
            __environments[__active_environment].append(func_id)
        return method(*args, **kwargs)

    return wrapper


def once_per_class(method):
    """
    The specified method will be invoked only once for all instances
    of a class within the environment.
    """
    def wrapper(*args, **kwargs):
        if __active_environment is None:
            activate_environment('default')

        func_id = repr(method.__name__) + repr(args[0].__class__)

        if func_id in __environments[__active_environment]:
            return
        else:
            __environments[__active_environment].append(func_id)
        return method(*args, **kwargs)

    return wrapper


def once_per_attribute_value(attr_name):
    """
    The specified method will be invoked once for all instances that share the
    same value for the specified attribute (sameness is established by comparing
    repr() of the values).
    """
    def wrapped_once_per_attribute_value(method):
        def wrapper(*args, **kwargs):
            if __active_environment is None:
                activate_environment('default')

            attr_value = getattr(args[0], attr_name)
            func_id = repr(method.__name__) + repr(args[0].__class__) + repr(attr_value)

            if func_id in __environments[__active_environment]:
                return
            else:
                __environments[__active_environment].append(func_id)
            return method(*args, **kwargs)

        return wrapper
    return wrapped_once_per_attribute_value


def once(method):
    """
    The specified method will be invoked only once within the
    environment.
    """
    def wrapper(*args, **kwargs):
        if __active_environment is None:
            activate_environment('default')

        func_id = repr(method.__code__)

        if func_id in __environments[__active_environment]:
            return
        else:
            __environments[__active_environment].append(func_id)
        return method(*args, **kwargs)

    return wrapper
