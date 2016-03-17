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
This module contains utilities for generating user documentation for Workload
Automation Plugins.

"""
import re
import inspect


PARAGRAPH_SEP = re.compile(r'\n\n+')
LINE_START = re.compile(r'\n\s*')


def get_paragraphs(text):
    """returns a list of paragraphs contained in the text"""
    return [LINE_START.sub(' ', p) for p in PARAGRAPH_SEP.split(text)]


class PluginDocumenter(object):

    @property
    def name(self):
        return self.ext.name

    @property
    def summary(self):
        """Returns the summary description for this Plugin, which, by
        convention, is the first paragraph of the description."""
        return get_paragraphs(self.description)[0]

    @property
    def description(self):
        """
        The description for an plugin is specified in the ``description``
        attribute, or (legacy) as a docstring for the plugin's class. If
        neither method is used in the Plugin, an empty string is returned.

        Description is assumed to be formed as reStructuredText. Leading and
        trailing whitespace will be stripped away.

        """
        if hasattr(self.ext, 'description'):
            return self.ext.description.strip()
        elif self.ext.__class__.__doc__:
            return self.ext.__class__.__doc__.strip()
        else:
            return ''

    @property
    def parameters(self):
        return [PluginParameterDocumenter(p) for p in self.ext.parameters]

    def __init__(self, ext):
        self.ext = ext


class PluginParameterDocumenter(object):

    @property
    def name(self):
        return self.param.name

    @property
    def kind(self):
        return self.param.get_type_name()

    @property
    def default(self):
        return self.param.default

    @property
    def description(self):
        return self.param.description

    @property
    def constraint(self):
        constraints = []
        if self.param.allowed_values:
            constraints.append('value must be in {}'.format(self.param.allowed_values))
        if self.param.constraint:
            constraint_text = self.param.constraint.__name__
            if constraint_text == '<lambda>':
                constraint_text = _parse_lambda(inspect.getsource(self.param.constraint))
            constraints.append(constraint_text)
        return ' and '.join(constraints)

    def __init__(self, param):
        self.param = param


# Utility functions


def _parse_lambda(text):
    """Parse the definition of a lambda function in to a readable string."""
    text = text.split('lambda')[1]
    param, rest = text.split(':')
    param = param.strip()
    # There are three things that could terminate a lambda: an (unparenthesized)
    # comma, a new line and an (unmatched) close paren.
    term_chars = [',', '\n', ')']
    func_text = ''
    inside_paren = 0  # an int rather than a bool to keep track of nesting
    for c in rest:
        if c in term_chars and not inside_paren:
            break
        elif c == ')':  # must be inside paren
            inside_paren -= 1
        elif c == '(':
            inside_paren += 1
        func_text += c

    # Rename the lambda parameter to 'value' so that the resulting
    # "description" makes more sense.
    func_text = re.sub(r'\b{}\b'.format(param), 'value', func_text)

    return func_text

