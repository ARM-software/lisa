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

import os
import subprocess

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
        import_error_str = import_error.message

if IPython and (IPython.version_info[0] == 2):
    import IPython.kernel
    import IPython.nbformat.v3

    def read_notebook(notebook_in):
        return IPython.nbformat.v3.reads_json(notebook_in)

    def write_notebook(notebook, fout):
        IPython.nbformat.v3.nbjson.JSONWriter().write(notebook, fout)

    NotebookNode = IPython.nbformat.v3.NotebookNode

elif IPython:
    # Unsupported IPython version
    IPython_ver_str = ".".join([str(n) for n in IPython.version_info])
    import_error_str = 'Unsupported IPython version {}'.format(IPython_ver_str)


def run_cell(kernel_client, cell):
    """Run a cell of a notebook in an ipython kernel and return its output"""
    kernel_client.execute(cell.input)

    outs = []
    while True:
        msg = kernel_client.get_iopub_msg()

        msg_type = msg["msg_type"]
        content = msg["content"]
        out = NotebookNode(output_type=msg_type)

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
            raise ValueError("Unknown msg_type {}".format(msg_type))

        outs.append(out)

    return outs


def run_notebook(notebook):
    """Run the notebook"""

    kernel_manager = IPython.kernel.KernelManager()
    kernel_manager.start_kernel(stderr=open(os.devnull, 'w'))
    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    for sheet in notebook.worksheets:
        for (prompt_number, cell) in enumerate(sheet.cells, 1):
            if cell.cell_type != "code":
                continue

            cell.outputs = run_cell(kernel_client, cell)

            cell.prompt_number = prompt_number
            if cell.outputs:
                cell.outputs[0]["prompt_number"] = prompt_number

    kernel_manager.shutdown_kernel()


def generate_pdf(nbbasename, output_directory):
    """Generate a PDF from the ipython notebook

    ipython nbconvert claims that the CLI is not stable, so keep this
    function here to be able to cope with inconsistencies

    """

    prev_dir = os.getcwd()
    os.chdir(output_directory)

    ipython_nbconvert = ['ipython', 'nbconvert', '--to=latex', '--post=PDF',
                         nbbasename]

    with open(os.devnull, 'w') as devnull:
        subprocess.check_call(ipython_nbconvert, stderr=devnull)

    os.chdir(prev_dir)

