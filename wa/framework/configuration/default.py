from wa.framework.configuration.core import MetaConfiguration, RunConfiguration
from wa.framework.configuration.plugin_cache import PluginCache
from wa.utils.serializer import yaml
from wa.utils.doc import strip_inlined_text

DEFAULT_INSTRUMENTS = ['execution_time',
                       'interrupts',
                       'cpufreq',
                       'status',
                       'standard',
                       'csv']


def _format_yaml_comment(param, short_description=False):
    comment = param.description
    comment = strip_inlined_text(comment)
    if short_description:
        comment = comment.split('\n\n')[0]
    comment = comment.replace('\n', '\n# ')
    comment = "# {}\n".format(comment)
    return comment


def _format_instruments(output):
    plugin_cache = PluginCache()
    output.write("instruments:\n")
    for plugin in DEFAULT_INSTRUMENTS:
        plugin_cls = plugin_cache.loader.get_plugin_class(plugin)
        output.writelines(_format_yaml_comment(plugin_cls, short_description=True))
        output.write(" - {}\n".format(plugin))
        output.write("\n")


def generate_default_config(path):
    with open(path, 'w') as output:
        for param in MetaConfiguration.config_points + RunConfiguration.config_points:
            entry = {param.name: param.default}
            comment = _format_yaml_comment(param)
            output.writelines(comment)
            yaml.dump(entry, output, default_flow_style=False)
            output.write("\n")
        _format_instruments(output)
