#! /usr/bin/env python3

import argparse

from lisa.conf import MultiSrcConf

def main():
    parser = argparse.ArgumentParser('Interpolate a LISA YAML configuration')
    parser.add_argument('conf', nargs='+', help='Target configuration files')

    args = parser.parse_args()
    conf_path_list = args.conf

    conf_map = MultiSrcConf.from_yaml_map_list(conf_path_list, add_default_src=False)

    def key(cls_and_conf):
        cls = cls_and_conf[0]
        return cls.__module__ + '.' + cls.__qualname__

    fragments = {}
    for conf_cls, conf in sorted(conf_map.items(), key=key):
        content = conf.to_yaml_map_str()
        fragments[conf_cls.__qualname__] = content

    print('\n'.join(
        '\n{}'.format(content)
        for name, content in sorted(fragments.items())
    ))

if __name__ == '__main__':
    main()
