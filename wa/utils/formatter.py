#    Copyright 2013-2017 ARM Limited
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


from wa.utils.terminalsize import get_terminal_size


INDENTATION_FROM_TITLE = 4


class TextFormatter(object):

    """
    This is a base class for text formatting. It mainly ask to implement two
    methods which are add_item and format_data. The formar will add new text to
    the formatter, whereas the latter will return a formatted text. The name
    attribute represents the name of the foramtter.
    """

    name = None
    data = None

    def __init__(self):
        pass

    def add_item(self, new_data, item_title):
        """
        Add new item to the text formatter.

        :param new_data: The data to be added
        :param item_title: A title for the added data
        """
        raise NotImplementedError()

    def format_data(self):
        """
        It returns a formatted text
        """
        raise NotImplementedError()


class DescriptionListFormatter(TextFormatter):

    name = 'description_list_formatter'
    data = None

    def get_text_width(self):
        if not self._text_width:
            self._text_width, _ = get_terminal_size()  # pylint: disable=unpacking-non-sequence
        return self._text_width

    def set_text_width(self, value):
        self._text_width = value

    text_width = property(get_text_width, set_text_width)

    def __init__(self, title=None, width=None):
        super(DescriptionListFormatter, self).__init__()
        self.data_title = title
        self._text_width = width
        self.longest_word_length = 0
        self.data = []

    def add_item(self, new_data, item_title):
        if len(item_title) > self.longest_word_length:
            self.longest_word_length = len(item_title)
        self.data[len(self.data):] = [(item_title, self._remove_newlines(new_data))]

    def format_data(self):
        parag_indentation = self.longest_word_length + INDENTATION_FROM_TITLE
        string_formatter = '{}:<{}{} {}'.format('{', parag_indentation, '}', '{}')

        formatted_data = ''
        if self.data_title:
            formatted_data += self.data_title

        line_width = self.text_width - parag_indentation
        for title, paragraph in self.data:
            formatted_data += '\n'
            title_len = self.longest_word_length - len(title)
            title += ':'
            if title_len > 0:
                title = (' ' * title_len) + title

            parag_lines = self._break_lines(paragraph, line_width).splitlines()
            if parag_lines:
                formatted_data += string_formatter.format(title, parag_lines[0])
                for line in parag_lines[1:]:
                    formatted_data += '\n' + string_formatter.format('', line)
            else:
                formatted_data += title[:-1]

        self.text_width = None
        return formatted_data

    # Return text's paragraphs sperated in a list, such that each index in the
    # list is a single text paragraph with no new lines
    def _remove_newlines(self, new_data):  # pylint: disable=R0201
        parag_list = ['']
        parag_num = 0
        prv_parag = None
        # For each paragraph sperated by a new line
        for paragraph in new_data.splitlines():
            if paragraph:
                parag_list[parag_num] += ' ' + paragraph
            # if the previous line is NOT empty, then add new empty index for
            # the next paragraph
            elif prv_parag:
                parag_num = 1
                parag_list.append('')
            prv_parag = paragraph

        # sometimes, we end up with an empty string as the last item so we reomve it
        if not parag_list[-1]:
            return parag_list[:-1]
        return parag_list

    def _break_lines(self, parag_list, line_width):  # pylint: disable=R0201
        formatted_paragraphs = []
        for para in parag_list:
            words = para.split()
            if words:
                formatted_text = words.pop(0)
                current_width = len(formatted_text)
                # for each word in the paragraph, line width is an accumlation of
                # word length + 1 (1 is for the space after each word).
                for word in words:
                    word = word.strip()
                    if current_width + len(word) + 1 >= line_width:
                        formatted_text += '\n' + word
                        current_width = len(word)
                    else:
                        formatted_text += ' ' + word
                        current_width += len(word) + 1
                formatted_paragraphs.append(formatted_text)
        return '\n\n'.join(formatted_paragraphs)
