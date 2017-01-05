#    Copyright 2015-2017 ARM Limited
#    Copyright 2016 Google Inc. All Rights Reserved.
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
"""Preprocessor to remove Marked Lines from IPython Output Cells"""


from nbconvert.exporters.html import HTMLExporter
from nbconvert.preprocessors import Preprocessor
import os
import re

REMOVE_START = '/* TRAPPY_PUBLISH_REMOVE_START */'
REMOVE_STOP = '/* TRAPPY_PUBLISH_REMOVE_STOP */'
REMOVE_LINE = '/* TRAPPY_PUBLISH_REMOVE_LINE */'
IMPORT_SCRIPT = r'/\* TRAPPY_PUBLISH_IMPORT = "([^"]+)" \*/'
SOURCE_LIB = r'<!-- TRAPPY_PUBLISH_SOURCE_LIB = "([^"]+)" -->'


class HTML(HTMLExporter):
    """HTML Exporter class for TRAPpy notebooks"""

    def __init__(self, **kwargs):
        super(HTML, self).__init__(**kwargs)
        self.register_preprocessor(TrappyPlotterPreprocessor, enabled=True)


class TrappyPlotterPreprocessor(Preprocessor):
    """Preprocessor to remove Marked Lines from IPython Output Cells"""

    def __init__(self, *args, **kwargs):
        super(Preprocessor, self).__init__(*args, **kwargs)
        self.inlined_files = []
        self.sourced_libs = []

    def preprocess_cell(self, cell, resources, cell_index):
        """Check if cell has text/html output and filter it"""

        if cell.cell_type == 'code' and hasattr(cell, "outputs"):
            for output in cell.outputs:
                if output.output_type == "display_data" and \
                   hasattr( output.data, "text/html"):
                    filtered =  self.filter_output(output.data["text/html"])
                    output.data["text/html"] = filtered
        return cell, resources

    def filter_output(self, output):
        """Function to remove marked lines"""

        lines = output.split('\n')

        final_lines = []
        multi_line_remove = False
        for line in lines:
            if REMOVE_START in line:
                multi_line_remove = True
                continue
            if REMOVE_STOP in line:
                multi_line_remove = False
                continue
            if multi_line_remove or REMOVE_LINE in line:
                continue

            import_match = re.search(IMPORT_SCRIPT, line)
            if import_match:
                trappy_base = os.path.dirname(os.path.dirname(__file__))
                import_file = os.path.join(trappy_base, import_match.group(1))
                if import_file in self.inlined_files:
                    continue

                with open(import_file) as fin:
                    final_lines.extend([l[:-1] for l in fin.readlines()])

                self.inlined_files.append(import_file)
                continue

            source_match = re.search(SOURCE_LIB, line)
            if source_match:
                lib_url = source_match.group(1)
                if lib_url in self.sourced_libs:
                    continue

                scl = '<script src="{}" type="text/javascript" charset="utf-8"></script>'.\
                      format(lib_url)
                final_lines.append(scl)

                self.sourced_libs.append(lib_url)
                continue

            final_lines.append(line)

        return '\n'.join(final_lines)
