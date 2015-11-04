#!/bin/bash

echo
echo "This script is required only for the development version"
echo "It will not be required once we have a stable vetsion".
echo "At that time everythin will work magically out-of-the-box"
echo

echo "Initializing dependency libraries..."
git submodule init
git submodule update

echo
echo
echo "NOTE: source the local init_env script before staring to use the suite."
echo "      This is required, for the time being, to proprely setup"
echo "      the PYTHONPATH for all the dependencies under libs"
echo
echo
echo "=== EAS RFC Test suite"
echo "To run the EAS RFC regression suite, first edit you configuration:"
echo "  target.config        : to setup your target board"
echo "  tests/eas/rfc.config : to define the experiment to run"
echo
echo "Than run the tests with:"
echo "  nosetests -v tests/eas/rfc.py"
echo
echo "Once tests have completed, you could report results by running"
echo "./tools/report.py"
echo
echo
echo "=== IPython Notebooks"
echo "An example ipython notbeook is provided under the ipynb folder"
echo "This folder contains also the script to properly start/stop"
echo "the IPython server."
echo "NOTE1: source the init_env before starting the server"
echo "NOTE2: start the server from withint the ipynb folder"
echo
echo
