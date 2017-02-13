from wlauto.core.configuration.configuration import (RunConfiguration,
                                                     JobGenerator, settings)
from wlauto.core.configuration.parsers import ConfigParser
from wlauto.core.configuration.plugin_cache import PluginCache


class WAState(object):
    """
    Represents run-time state of WA. Mostly used as a container for loaded 
    configuration and discovered plugins.

    This exists outside of any command or run and is associated with the running 
    instance of wA itself.
    """

    def __init__(self, settings=settings):
        self.settings = settings
        self.run_config = RunConfiguration()
        self.plugin_cache = PluginCache()
        self.jobs_config = JobGenerator(self.plugin_cache)
        self.loaded_config_sources = []
        self._config_parser = ConfigParser()

    def load_config_file(self, filepath):
        self._config_parser.load_from_path(self, filepath)
        self.loaded_config_sources.append(filepath)

    def load_config(self, values, source, wrap_exceptions=True):
        self._config_parser.load(self, values, source)
        self.loaded_config_sources.append(source)


