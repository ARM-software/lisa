import sys
from collections import OrderedDict

from wa import ComplexCommand, SubCommand, pluginloader
from wa.framework.target.descriptor import get_target_descriptions
from wa.framework.exception import ConfigError

from wa.utils.serializer import yaml


class CreateAgendaSubcommand(SubCommand):

    name = 'agenda'
    description = """
    Create an agenda with the specified extensions enabled. And parameters set
    to their default values.
    """

    def initialize(self, context):
        self.parser.add_argument('plugins', nargs='+',
                                 help='Plugins to be added to the agendas')
        self.parser.add_argument('-i', '--iterations', type=int, default=1,
                                 help='Sets the number of iterations for all workloads')
        self.parser.add_argument('-o', '--output', metavar='FILE',
                                 help='Output file. If not specfied, STDOUT will be used instead.')

    def execute(self, state, args):
        agenda = OrderedDict()
        agenda['config'] = OrderedDict(instrumentation=[], result_processors=[])
        agenda['global'] = OrderedDict(iterations=args.iterations)
        agenda['workloads'] = []
        target_desc = None

        targets = {td.name: td for td in get_target_descriptions()}

        for name in args.plugins:
            if name in targets:
                if target_desc is not None:
                    raise ConfigError('Specifying multiple devices: {} and {}'.format(target_desc.name, name))
                target_desc = targets[name]
                agenda['config']['device'] = name
                agenda['config']['device_config'] = target_desc.get_default_config()
                continue

            extcls = pluginloader.get_plugin_class(name)
            config = pluginloader.get_default_config(name)

            if extcls.kind == 'workload':
                entry = OrderedDict()
                entry['name'] = extcls.name
                if name != extcls.name:
                    entry['label'] = name
                entry['params'] = config
                agenda['workloads'].append(entry)
            else:
                if extcls.kind == 'instrument':
                    agenda['config']['instrumentation'].append(name)
                if extcls.kind == 'result_processor':
                    agenda['config']['result_processors'].append(name)
                agenda['config'][name] = config

        if args.output:
            wfh = open(args.output, 'w')
        else:
            wfh = sys.stdout
        yaml.dump(agenda, wfh, indent=4, default_flow_style=False)
        if args.output:
            wfh.close()


class CreateCommand(ComplexCommand):

    name = 'create'
    description = '''
    Used to create various WA-related objects (see positional arguments list
    for what objects may be created).\n\nUse "wa create <object> -h" for
    object-specific arguments.
    '''
    subcmd_classes = [
        #CreateWorkloadSubcommand,
        #CreatePackageSubcommand,
        CreateAgendaSubcommand,
    ]
