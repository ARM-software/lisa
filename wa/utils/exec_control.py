# "environment" management:
__environments = {}
__active_environment = None


def activate_environment(name):
    """
    Sets the current tracking environment to ``name``. If an
    environment with that name does not already exist, it will be
    created.
    """
    #pylint: disable=W0603
    global __active_environment

    if name not in __environments.keys():
        init_environment(name)
    __active_environment = name

def init_environment(name):
    """
    Create a new environment called ``name``, but do not set it as the
    current environment.

    :raises: ``ValueError`` if an environment with name ``name``
             already exists.
    """
    if name in __environments.keys():
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
        if name not in __environments.keys():
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
        func_id = repr(method.__hash__()) + repr(args[0])
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

        func_id = repr(method.func_name) + repr(args[0].__class__)

        if func_id in __environments[__active_environment]:
            return
        else:
            __environments[__active_environment].append(func_id)
        return method(*args, **kwargs)

    return wrapper

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
