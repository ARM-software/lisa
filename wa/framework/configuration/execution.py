from copy import copy
from collections import OrderedDict

from wa.framework import pluginloader
from wa.framework.exception import ConfigError
from wa.framework.configuration.core import ConfigurationPoint
from wa.framework.utils.types import TreeNode, list_of, identifier


class ExecConfig(object):

    static_config_points = [
            ConfigurationPoint(
                'components',
                kind=list_of(identifier),
                description="""
                Components to be activated.
                """,
            ),
            ConfigurationPoint(
                'runtime_parameters',
                kind=list_of(identifier),
                aliases=['runtime_params'],
                description="""
                Components to be activated.
                """,
            ),
            ConfigurationPoint(
                'classifiers',
                kind=list_of(str),
                description="""
                Classifiers to be used. Classifiers are arbitrary key-value
                pairs associated with with config. They may be used during output
                proicessing and should be used to provide additional context for
                collected results.
                """,
            ),
    ]

    config_points = None

    @classmethod
    def _load(cls, load_global=False, loader=pluginloader):
        if cls.config_points is None:
            cls.config_points = {c.name: c for c in cls.static_config_points}
            for plugin in loader.list_plugins():
                cp = ConfigurationPoint(
                    plugin.name,
                    kind=OrderedDict,
                    description="""
                    Configuration for {} plugin.
                    """.format(plugin.name)
                )
                cls._add_config_point(plugin.name, cp)
                for alias in plugin.aliases:
                    cls._add_config_point(alias.name, cp)

    @classmethod
    def _add_config_point(cls, name, cp):
        if name in cls.config_points:
            message = 'Cofig point for "{}" already exists ("{}")'
            raise ValueError(message.format(name, cls.config_points[name].name))



class GlobalExecConfig(ExecConfig):

