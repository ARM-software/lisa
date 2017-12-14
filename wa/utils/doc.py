#    Copyright 2014-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
Utilities for working with and formatting documentation.

"""
import os
import re
import inspect
from itertools import cycle

USER_HOME = os.path.expanduser('~')

BULLET_CHARS = '-*'


def get_summary(aclass):
    """
    Returns the summary description for an extension class. The summary is the
    first paragraph (separated by blank line) of the description taken either from
    the ``descripton`` attribute of the class, or if that is not present, from the
    class' docstring.

    """
    return get_description(aclass).split('\n\n')[0]


def get_description(aclass):
    """
    Return the description of the specified extension class. The description is taken
    either from ``description`` attribute of the class or its docstring.

    """
    if hasattr(aclass, 'description') and aclass.description:
        return inspect.cleandoc(aclass.description)
    if aclass.__doc__:
        return inspect.getdoc(aclass)
    else:
        return 'no documentation found for {}'.format(aclass.__name__)


def get_type_name(obj):
    """Returns the name of the type object or function specified. In case of a lambda,
    the definiition is returned with the parameter replaced by "value"."""
    match = re.search(r"<(type|class|function) '?(.*?)'?>", str(obj))
    if isinstance(obj, tuple):
        name = obj[1]
    elif match.group(1) == 'function':
        text = str(obj)
        name = text.split()[1]
        if name == '<lambda>':
            source = inspect.getsource(obj).strip().replace('\n', ' ')
            match = re.search(r'lambda\s+(\w+)\s*:\s*(.*?)\s*[\n,]', source)
            if not match:
                raise ValueError('could not get name for {}'.format(obj))
            name = match.group(2).replace(match.group(1), 'value')
    else:
        name = match.group(2)
        if '.' in name:
            name = name.split('.')[-1]
    return name


def count_leading_spaces(text):
    """
    Counts the number of leading space characters in a string.

    TODO: may need to update this to handle whitespace, but shouldn't
          be necessary as there should be no tabs in Python source.

    """
    nspaces = 0
    for c in text:
        if c == ' ':
            nspaces += 1
        else:
            break
    return nspaces


def format_column(text, width):
    """
    Formats text into a column of specified width. If a line is too long,
    it will be broken on a word boundary. The new lines will have the same
    number of leading spaces as the original line.

    Note: this will not attempt to join up lines that are too short.

    """
    formatted = []
    for line in text.split('\n'):
        line_len = len(line)
        if line_len <= width:
            formatted.append(line)
        else:
            words = line.split(' ')
            new_line = words.pop(0)
            while words:
                next_word = words.pop(0)
                if (len(new_line) + len(next_word) + 1) < width:
                    new_line += ' ' + next_word
                else:
                    formatted.append(new_line)
                    new_line = ' ' * count_leading_spaces(new_line) + next_word
            formatted.append(new_line)
    return '\n'.join(formatted)


def format_bullets(text, width, char='-', shift=3, outchar=None):
    """
    Formats text into bulleted list. Assumes each line of input that starts with
    ``char`` (possibly preceeded with whitespace) is a new bullet point. Note: leading
    whitespace in the input will *not* be preserved. Instead, it will be determined by
    ``shift`` parameter.

    :text: the text to be formated
    :width: format width (note: must be at least ``shift`` + 4).
    :char: character that indicates a new bullet point in the input text.
    :shift: How far bulleted entries will be indented. This indicates the indentation
            level of the bullet point. Text indentation level will be ``shift`` + 3.
    :outchar: character that will be used to mark bullet points in the output. If
              left as ``None``, ``char`` will be used.

    """
    bullet_lines = []
    output = ''

    def __process_bullet(bullet_lines):
        if bullet_lines:
            bullet = format_paragraph(indent(' '.join(bullet_lines), shift + 2), width)
            bullet = bullet[:3] + outchar + bullet[4:]
            del bullet_lines[:]
            return bullet + '\n'
        else:
            return ''

    if outchar is None:
        outchar = char
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith(char):  # new bullet
            output += __process_bullet(bullet_lines)
            line = line[1:].strip()
        bullet_lines.append(line)
    output += __process_bullet(bullet_lines)
    return output


def format_simple_table(rows, headers=None, align='>', show_borders=True, borderchar='='):  # pylint: disable=R0914
    """Formats a simple table."""
    if not rows:
        return ''
    rows = [map(str, r) for r in rows]
    num_cols = len(rows[0])

    # cycle specified alignments until we have num_cols of them. This is
    # consitent with how such cases are handled in R, pandas, etc.
    it = cycle(align)
    align = [it.next() for _ in xrange(num_cols)]

    cols = zip(*rows)
    col_widths = [max(map(len, c)) for c in cols]
    if headers:
        col_widths = [max(len(h), cw) for h, cw in zip(headers, col_widths)]
    row_format = ' '.join(['{:%s%s}' % (align[i], w) for i, w in enumerate(col_widths)])
    row_format += '\n'

    border = row_format.format(*[borderchar * cw for cw in col_widths])

    result = border if show_borders else ''
    if headers:
        result += row_format.format(*headers)
        result += border
    for row in rows:
        result += row_format.format(*row)
    if show_borders:
        result += border
    return result


def format_paragraph(text, width):
    """
    Format the specified text into a column of specified with. The text is
    assumed to be a single paragraph and existing line breaks will not be preserved.
    Leading spaces (of the initial line), on the other hand, will be preserved.

    """
    text = re.sub('\n\n*\\s*', ' ', text.strip('\n'))
    return format_column(text, width)


def format_body(text, width):
    """
    Format the specified text into a column  of specified width. The text is
    assumed to be a "body" of one or more paragraphs separated by one or more
    blank lines. The initial indentation of the first line of each paragraph
    will be presevered, but any other formatting may be clobbered.

    """
    text = re.sub('\n\\s*\n', '\n\n', text.strip('\n'))  # get rid of all-whitespace lines
    paragraphs = re.split('\n\n+', text)
    formatted_paragraphs = []
    for p in paragraphs:
        if p.strip() and p.strip()[0] in BULLET_CHARS:
            formatted_paragraphs.append(format_bullets(p, width))
        else:
            formatted_paragraphs.append(format_paragraph(p, width))
    return '\n\n'.join(formatted_paragraphs)


def strip_inlined_text(text):
    """
    This function processes multiline inlined text (e.g. form docstrings)
    to strip away leading spaces and leading and trailing new lines.

    """
    text = text.strip('\n')
    lines = [ln.rstrip() for ln in text.split('\n')]

    # first line is special as it may not have the indet that follows the
    # others, e.g. if it starts on the same as the multiline quote (""").
    nspaces = count_leading_spaces(lines[0])

    if len([ln for ln in lines if ln]) > 1:
        to_strip = min(count_leading_spaces(ln) for ln in lines[1:] if ln)
        if nspaces >= to_strip:
            stripped = [lines[0][to_strip:]]
        else:
            stripped = [lines[0][nspaces:]]
        stripped += [ln[to_strip:] for ln in lines[1:]]
    else:
        stripped = [lines[0][nspaces:]]
    return '\n'.join(stripped).strip('\n')


def indent(text, spaces=4):
    """Indent the lines i the specified text by ``spaces`` spaces."""
    indented = []
    for line in text.split('\n'):
        if line:
            indented.append(' ' * spaces + line)
        else:  # do not indent emtpy lines
            indented.append(line)
    return '\n'.join(indented)


def format_literal(lit):
    if isinstance(lit, basestring):
        return '``\'{}\'``'.format(lit)
    elif hasattr(lit, 'pattern'):  # regex
        return '``r\'{}\'``'.format(lit.pattern)
    else:
        return '``{}``'.format(lit)


def get_params_rst(parameters):
    text = ''
    for param in parameters:
        text += '{} : {} {}\n'.format(param.name, get_type_name(param.kind),
                                      param.mandatory and '(mandatory)' or ' ')
        desc = strip_inlined_text(param.description or '')
        text += indent('{}\n'.format(desc))
        if param.allowed_values:
            text += indent('\nallowed values: {}\n'.format(', '.join(map(format_literal, param.allowed_values))))
        elif param.constraint:
            text += indent('\nconstraint: ``{}``\n'.format(get_type_name(param.constraint)))
        if param.default:
            value = param.default
            if isinstance(value, basestring) and value.startswith(USER_HOME):
                value = value.replace(USER_HOME, '~')
            text += indent('\ndefault: {}\n'.format(format_literal(value)))
        text += '\n'
    return text


def underline(text, symbol='='):
    return '{}\n{}\n\n'.format(text, symbol * len(text))


def get_rst_from_plugin(plugin):
    text = underline(plugin.name, '-')
    if hasattr(plugin, 'description'):
        desc = strip_inlined_text(plugin.description or '')
    elif plugin.__doc__:
        desc = strip_inlined_text(plugin.__doc__)
    else:
        desc = ''
    text += desc + '\n\n'
    params_rst = get_params_rst(plugin.parameters)
    if params_rst:
        text += underline('parameters', '~') + params_rst
    return text + '\n'
