from wlauto.core.configuration.configuration import (MetaConfiguration,
                                                     RunConfiguration,
                                                     JobGenerator, settings)
from wlauto.core.configuration.parsers import ConfigParser
from wlauto.core.configuration.plugin_cache import PluginCache


class CombinedConfig(object):

    @staticmethod
    def from_pod(pod):
        instance = CombinedConfig()
        instance.settings = MetaConfiguration.from_pod(pod.get('setttings', {}))
        instance.run_config = RunConfiguration.from_pod(pod.get('run_config', {}))
        return instance

    def __init__(self, settings=None, run_config=None):
        self.settings = settings
        self.run_config = run_config

    def to_pod(self):
        return {'settings': self.settings.to_pod(),
                'run_config': self.run_config.to_pod()}


class ConfigManager(object):
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
        self._job_specs = []
        self.jobs = []

    def load_config_file(self, filepath):
        self._config_parser.load_from_path(self, filepath)
        self.loaded_config_sources.append(filepath)

    def load_config(self, values, source, wrap_exceptions=True):
        self._config_parser.load(self, values, source)
        self.loaded_config_sources.append(source)

    def finalize(self):
        self.run_config.merge_device_config(self.plugin_cache)
        return CombinedConfig(self.settings, self.run_config)

