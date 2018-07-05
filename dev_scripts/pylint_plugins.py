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

                text = text.split('\n')
                # NOTE: doing it this way because the "correct" approach below does not
                #       work. We can get away with this, because in well-formated WA files,
                #       the initial line is the copyright header's blank line.
                if 'pylint:' in text[0]:
                    msg = 'pylint directive found on the first line of {}; please move to below copyright header'
                    raise RuntimeError(msg.format(mod.name))
                if text[0].strip() and text[0][0] != '#':
                    msg = 'first line of {} is not a comment; is the copyright header missing?'
                    raise RuntimeError(msg.format(mod.name))
                text[0] = '# pylint: disable={}'.format(','.join(errors))
                mod.file_bytes = '\n'.join(text)

                # This is what *should* happen, but doesn't work.
                # text.insert(0, '# pylint: disable=attribute-defined-outside-init')
                # mod.file_bytes = '\n'.join(text)
                # mod.tolineno += 1


MANAGER.register_transform(scoped_nodes.Module, transform)
