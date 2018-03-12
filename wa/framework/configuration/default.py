from wa.framework.configuration.core import MetaConfiguration, RunConfiguration
from wa.framework.configuration.plugin_cache import PluginCache
from wa.utils.serializer import yaml
from wa.utils.doc import strip_inlined_text

DEFAULT_AUGMENTATIONS = ['execution_time',
                         'interrupts',
                         'cpufreq',
                         'status',
                         'csv'
                        ]


def _format_yaml_comment(param, short_description=False):
    comment = param.description
    comment = strip_inlined_text(comment)
    if short_description:
        comment = comment.split('\n\n')[0]
    comment = comment.replace('\n', '\n# ')
    comment = "# {}\n".format(comment)
    return comment


def _format_augmentations(output):
    plugin_cache = PluginCache()
    output.write("augmentations:\n")
    for plugin in DEFAULT_AUGMENTATIONS:
        plugin_cls = plugin_cache.loader.get_plugin_class(plugin)
        output.writelines(_format_yaml_comment(plugin_cls, short_description=True))
        output.write(" - {}\n".format(plugin))
        output.write("\n")


def generate_default_config(path):
    with open(path, 'w') as output:
        for param in MetaConfiguration.config_points + RunConfiguration.config_points:
            entry = {param.name: param.default}
            write_param_yaml(entry, param, output)
        _format_augmentations(output)


def write_param_yaml(entry, param, output):
    comment = _format_yaml_comment(param)
    output.writelines(comment)
    yaml.dump(entry, output, default_flow_style=False)
    output.write("\n")
