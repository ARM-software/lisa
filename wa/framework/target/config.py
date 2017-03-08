from copy import copy

#Not going to be used for now.

class TargetConfig(dict):
    """
    Represents a configuration for a target.

    """
    def __init__(self, config=None):
        if isinstance(config, TargetConfig):
            self.__dict__ = copy(config.__dict__)
        elif hasattr(config, 'iteritems'):
            for k, v in config.iteritems:
                self.set(k, v)
        elif config:
            raise ValueError(config)

    def set(self, name, value):
        setattr(self, name, value)
