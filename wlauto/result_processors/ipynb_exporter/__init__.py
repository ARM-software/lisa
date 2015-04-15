#    Copyright 2015 ARM Limited
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

# pylint: disable=attribute-defined-outside-init

from datetime import datetime
import os
import shutil
import subprocess
import webbrowser

from wlauto import File, Parameter, ResultProcessor
from wlauto.exceptions import ConfigError, ResultProcessorError
from wlauto.utils.misc import open_file

try:
    import IPython.kernel as ipython_kernel
    import IPython.nbformat.current as ipython_nbformat
except ImportError:
    ipython_kernel = None
    ipython_nbformat = None

try:
    import jinja2
except ImportError:
    jinja2 = None

DEFAULT_NOTEBOOK_TEMPLATE = 'template.ipynb'


class IPythonNotebookExporter(ResultProcessor):

    name = 'ipynb_exporter'
    description = """
    Generates an IPython notebook from a template with the results and runs it.
    Optionally it can show the resulting notebook in a web browser.
    It can also generate a PDF from the notebook.

    The template syntax is that of `jinja2 <http://jinja.pocoo.org/>`_
    and the template should generate a valid ipython notebook. The
    templates receives ``result`` and ``context`` which correspond to
    the RunResult and ExecutionContext respectively. You can use those
    in your ipython notebook template to extract any information you
    want to parse or show.

    This results_processor depends on ``ipython`` and ``python-jinja2`` being
    installed on the system.

    For example, a simple template that plots a bar graph of the results is::

    """
    # Note: the example template is appended after the class definition

    parameters = [
        Parameter('notebook_template', default=DEFAULT_NOTEBOOK_TEMPLATE,
                  description='''Filename of the ipython notebook template.  If
                  no `notebook_template` is specified, the example template
                  above is used.'''),
        Parameter('notebook_name_prefix', default='result_',
                  description=''' Prefix of the name of the notebook. The date,
                  time and ``.ipynb`` are appended to form the notebook filename.
                  E.g. if notebook_name_prefix is ``result_`` then a run on 13th
                  April 2015 at 9:54 would generate a notebook called
                  ``result_150413-095400.ipynb``. When generating a PDF,
                  the resulting file will have the same name, but
                  ending in ``.pdf``.'''),
        Parameter('show_notebook', kind=bool,
                  description='Open a web browser with the resulting notebook.'),
        Parameter('notebook_directory',
                  description='''Path to the notebooks directory served by the
                  ipython notebook server. You must set it if
                  ``show_notebook`` is selected. The ipython notebook
                  will be copied here if specified.'''),
        Parameter('notebook_url', default='http://localhost:8888/notebooks',
                  description='''URL of the notebook on the IPython server. If
                  not specified, it will be assumed to be in the root notebooks
                  location on localhost, served on port 8888. Only needed if
                  ``show_notebook`` is selected.

                  .. note:: the URL should not contain the final part (the notebook name) which will be populated automatically.
                  '''),
        Parameter('convert_to_pdf', kind=bool,
                  description='Convert the resulting notebook to PDF.'),
        Parameter('show_pdf', kind=bool,
                  description='''Open the pdf at the end of the run. This can
                  only be selected if convert_to_pdf has also been selected.'''),
    ]

    def initialize(self, context):
        file_resource = File(self, self.notebook_template)
        self.notebook_template_file = context.resolver.get(file_resource)
        nbbasename_template = self.notebook_name_prefix + '%y%m%d-%H%M%S.ipynb'
        self.nbbasename = datetime.now().strftime(nbbasename_template)

    def validate(self):
        if not ipython_kernel:
            msg = '{} requires ipython package to be installed'.format(self.name)
            raise ResultProcessorError(msg)

        if not jinja2:
            msg = '{} requires python-jinja2 package to be installed'.format(self.name)
            raise ResultProcessorError(msg)

        if self.show_notebook and not self.notebook_directory:
            raise ConfigError('Requested "show_notebook" but no notebook_directory was specified')

        if self.notebook_directory and not os.path.isdir(self.notebook_directory):
            raise ConfigError('notebook_directory {} does not exist'.format(self.notebook_directory))

        if self.show_pdf and not self.convert_to_pdf: #pylint: disable=E0203
            self.convert_to_pdf = True
            self.logger.debug('Assuming "convert_to_pdf" as "show_pdf" is set')

    def export_run_result(self, result, context):
        self.generate_notebook(result, context)
        if self.show_notebook:
            self.open_notebook()

        if self.convert_to_pdf:
            self.generate_pdf(context.run_output_directory)
            if self.show_pdf:
                self.open_pdf()

    def generate_notebook(self, result, context):
        """Generate a notebook from the template and run it"""
        with open(self.notebook_template_file) as fin:
            template = jinja2.Template(fin.read())

        notebook_in = template.render(result=result, context=context)
        notebook = ipython_nbformat.reads(notebook_in, 'json')

        self.run_notebook(notebook)

        self.notebook_file = os.path.join(context.run_output_directory,
                                          self.nbbasename)
        with open(self.notebook_file, 'w') as wfh:
            ipython_nbformat.write(notebook, wfh, 'json')

        if self.notebook_directory:
            shutil.copy(self.notebook_file,
                        os.path.join(self.notebook_directory))

    def run_notebook(self, notebook):
        """Run the notebook"""

        kernel_manager = ipython_kernel.KernelManager()
        kernel_manager.start_kernel(stderr=open(os.devnull, 'w'))
        kernel_client = kernel_manager.client()
        kernel_client.start_channels()

        for sheet in notebook.worksheets:
            for (prompt_number, cell) in enumerate(sheet.cells, 1):
                if cell.cell_type != "code":
                    continue

                cell.outputs = self.run_cell(kernel_client, cell)

                cell.prompt_number = prompt_number
                if cell.outputs:
                    cell.outputs[0]["prompt_number"] = prompt_number

        kernel_manager.shutdown_kernel()

    def run_cell(self, kernel_client, cell): # pylint: disable=R0201
        """Run a cell of a notebook in an ipython kernel and return its output"""
        kernel_client.shell_channel.execute(cell.input)

        outs = []
        while True:
            msg = kernel_client.get_iopub_msg()

            msg_type = msg["msg_type"]
            content = msg["content"]
            out = ipython_nbformat.NotebookNode(output_type=msg_type)

            if msg_type == "status":
                if content["execution_state"] == "idle":
                    break
                else:
                    continue
            elif msg_type == "pyin":
                continue
            elif msg_type == "stream":
                out.stream = content["name"]
                out.text = content["data"]
            elif msg_type in ("display_data", "pyout"):
                for mime, data in content["data"].iteritems():
                    if mime == "text/plain":
                        attr = "text"
                    else:
                        attr = mime.split("/")[-1]
                    setattr(out, attr, data)
            elif msg_type == "pyerr":
                out.ename = content["ename"]
                out.evalue = content["evalue"]
                out.traceback = content["traceback"]
            else:
                raise ValueError("Unknown msg_type %s".format(msg_type))

            outs.append(out)

        return outs

    def open_notebook(self):
        """Open the notebook in a browser"""
        webbrowser.open(self.notebook_url.rstrip('/') + '/' + self.nbbasename)

    def generate_pdf(self, output_directory):
        """Generate a PDF from the ipython notebook"""
        # ipython nbconvert generates the pdf and other temporary files in the
        # current directory.  Move to the results directory to avoid polluting a
        # random directory

        prev_dir = os.getcwd()
        os.chdir(output_directory)

        ipython_nbconvert = ['ipython', 'nbconvert', '--to=latex', '--post=PDF',
                             self.nbbasename]

        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(ipython_nbconvert, stderr=devnull)

        os.chdir(prev_dir)

    def open_pdf(self):
        """Open the PDF"""
        pdf_file = os.path.splitext(self.notebook_file)[0] + ".pdf"
        open_file(pdf_file)


# Add the default template to the documentation
with open(os.path.join(os.path.dirname(__file__), DEFAULT_NOTEBOOK_TEMPLATE)) as in_file:
    # Without an empty indented line, wlauto.misc.doc.strip_inlined_text() gets
    # confused
    IPythonNotebookExporter.description += "     \n"

    for line in in_file:
        IPythonNotebookExporter.description += "     " + line
