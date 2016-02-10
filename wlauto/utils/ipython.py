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
# pylint: disable=no-name-in-module,import-error,no-member

import os
import subprocess
from distutils.version import LooseVersion

# pylint: disable=wrong-import-position,ungrouped-imports
import_error_str = ''
try:
    import IPython
except ImportError as import_error:
    IPython = None
    # Importing IPython can fail for a variety of reasons, report the actual
    # failure unless it's just that the package is not present
    if import_error.message.startswith("No module named"):  # pylint: disable=E1101
        import_error_str = 'ipynb_exporter requires ipython package to be installed'
    else:
        import_error_str = import_error.message  # pylint: disable=redefined-variable-type

# The current code generates notebooks version 3
NBFORMAT_VERSION = 3


if IPython:
    if LooseVersion('5.0.0') > LooseVersion(IPython.__version__) >= LooseVersion('4.0.0'):
        import nbformat
        from jupyter_client.manager import KernelManager

        def read_notebook(notebook_in):  # pylint: disable=function-redefined
            return nbformat.reads(notebook_in, NBFORMAT_VERSION)  # pylint: disable=E1101

        def write_notebook(notebook, fout):  # pylint: disable=function-redefined
            nbformat.write(notebook, fout)  # pylint: disable=E1101

        NotebookNode = nbformat.NotebookNode  # pylint: disable=E1101

        IPYTHON_NBCONVERT_HTML = ['jupyter', 'nbconvert', '--to html']
        IPYTHON_NBCONVERT_PDF = ['jupyter', 'nbconvert', '--to pdf']

    elif LooseVersion('4.0.0') > LooseVersion(IPython.__version__) >= LooseVersion('3.0.0'):
        from IPython.kernel import KernelManager
        import IPython.nbformat

        def read_notebook(notebook_in):  # pylint: disable=function-redefined
            return IPython.nbformat.reads(notebook_in, NBFORMAT_VERSION)  # pylint: disable=E1101

        def write_notebook(notebook, fout):  # pylint: disable=function-redefined
            IPython.nbformat.write(notebook, fout)  # pylint: disable=E1101

        NotebookNode = IPython.nbformat.NotebookNode  # pylint: disable=E1101

        IPYTHON_NBCONVERT_HTML = ['ipython', 'nbconvert', '--to=html']
        IPYTHON_NBCONVERT_PDF = ['ipython', 'nbconvert', '--to=pdf']
    elif LooseVersion('3.0.0') > LooseVersion(IPython.__version__) >= LooseVersion('2.0.0'):
        from IPython.kernel import KernelManager
        import IPython.nbformat.v3

        def read_notebook(notebook_in):  # pylint: disable=function-redefined
            return IPython.nbformat.v3.reads_json(notebook_in)  # pylint: disable=E1101

        def write_notebook(notebook, fout):  # pylint: disable=function-redefined
            IPython.nbformat.v3.nbjson.JSONWriter().write(notebook, fout)  # pylint: disable=E1101

        NotebookNode = IPython.nbformat.v3.NotebookNode  # pylint: disable=E1101

        IPYTHON_NBCONVERT_HTML = ['ipython', 'nbconvert', '--to=html']
        IPYTHON_NBCONVERT_PDF = ['ipython', 'nbconvert', '--to=latex',
                                 '--post=PDF']
    else:
        # Unsupported IPython version
        import_error_str = 'Unsupported IPython version {}'.format(IPython.__version__)


def parse_valid_output(msg):
    """Parse a valid result from an execution of a cell in an ipython kernel"""
    msg_type = msg["msg_type"]
    if msg_type == 'error':
        msg_type = 'pyerr'
    elif msg_type == 'execute_result':
        msg_type = 'pyout'

    content = msg["content"]
    out = NotebookNode(output_type=msg_type)

    if msg_type == "stream":
        out.stream = content["name"]
        try:
            out.text = content['data']
        except KeyError:
            out.text = content['text']
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
        raise ValueError("Unknown msg_type {}".format(msg_type))

    return out


def run_cell(kernel_client, cell):
    """Run a cell of a notebook in an ipython kernel and return its output"""
    kernel_client.execute(cell.input)

    input_acknowledged = False
    outs = []
    while True:
        msg = kernel_client.get_iopub_msg()

        if msg["msg_type"] == "status":
            if msg["content"]["execution_state"] == "idle" and input_acknowledged:
                break
        elif msg["msg_type"] in ('pyin', 'execute_input'):
            input_acknowledged = True
        else:
            out = parse_valid_output(msg)
            outs.append(out)

    return outs


def run_notebook(notebook):
    """Run the notebook"""
    kernel_manager = KernelManager()
    kernel_manager.start_kernel(stderr=open(os.devnull, 'w'))
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    for sheet in notebook.worksheets:
        for (prompt_number, cell) in enumerate(sheet.cells, 1):
            if cell.cell_type != "code":
                continue

            cell.outputs = run_cell(kernel_client, cell)

            cell.prompt_number = prompt_number
            if cell.outputs and cell.outputs[0]['output_type'] == 'pyout':
                cell.outputs[0]["prompt_number"] = prompt_number

    kernel_manager.shutdown_kernel()


def export_notebook(nbbasename, output_directory, output_format):
    """Generate a PDF or HTML from the ipython notebook

    output_format has to be either 'pdf' or 'html'.  These are the
    only formats currently supported.

    ipython nbconvert claims that the CLI is not stable, so keep this
    function here to be able to cope with inconsistencies

    """

    if output_format == "html":
        ipython_command = IPYTHON_NBCONVERT_HTML
    elif output_format == "pdf":
        ipython_command = IPYTHON_NBCONVERT_PDF
    else:
        raise ValueError("Unknown output format: {}".format(output_format))

    prev_dir = os.getcwd()
    os.chdir(output_directory)

    with open(os.devnull, 'w') as devnull:
        subprocess.check_call(ipython_command + [nbbasename], stderr=devnull)

    os.chdir(prev_dir)
