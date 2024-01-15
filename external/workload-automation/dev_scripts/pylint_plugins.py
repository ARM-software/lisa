import sys

from astroid import MANAGER
from astroid import scoped_nodes


IGNORE_ERRORS = {
        ('attribute-defined-outside-init', ): [
            'wa.workloads',
            'wa.instruments',
            'wa.output_procesors',
        ]
}


def register(linter):
    pass


def transform(mod):
    for errors, paths in IGNORE_ERRORS.items():
        for path in paths:
            if path in mod.name:
                text = mod.stream().read()
                if not text.strip():
                    return

                text = text.split(b'\n')
                # NOTE: doing it this way because the "correct" approach below does not
                #       work. We can get away with this, because in well-formated WA files,
                #       the initial line is the copyright header's blank line.
                if b'pylint:' in text[0]:
                    msg = 'pylint directive found on the first line of {}; please move to below copyright header'
                    raise RuntimeError(msg.format(mod.name))
                char = chr(text[0][0])
                if text[0].strip() and char != '#':
                    msg = 'first line of {} is not a comment; is the copyright header missing?'
                    raise RuntimeError(msg.format(mod.name))
                text[0] = '# pylint: disable={}'.format(','.join(errors)).encode('utf-8')
                mod.file_bytes = b'\n'.join(text)

                # This is what *should* happen, but doesn't work.
                # text.insert(0, '# pylint: disable=attribute-defined-outside-init')
                # mod.file_bytes = '\n'.join(text)
                # mod.tolineno += 1


MANAGER.register_transform(scoped_nodes.Module, transform)
