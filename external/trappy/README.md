TRAPpy [![Build Status](https://travis-ci.org/ARM-software/trappy.svg?branch=master)](https://travis-ci.org/ARM-software/trappy) [![Version](https://img.shields.io/pypi/v/trappy.svg)](https://pypi.python.org/pypi/trappy)
======

TRAPpy (Trace Analysis and Plotting in Python) is a visualization tool to help
analyze data generated on a device. It parses ftrace-like logs and creates
in-memory data structures to be used for plotting and data analysis.

# Installation

The following instructions are for Ubuntu 14.04 LTS but they should
also work with Debian jessie.  Older versions of Ubuntu or Debian
(e.g. Ubuntu 12.04 or Debian wheezy) will likely require to install
more packages from pip as the ones present in Ubuntu 12.04 or Debian
wheezy will probably be too old.

## Required dependencies

##### Install additional tools required for some tests and functionalities

	$ sudo apt install trace-cmd kernelshark

##### Install the Python package manager

	$ sudo apt install python-pip python-dev

##### Install required python packages

	$ sudo apt install libfreetype6-dev libpng12-dev python-nose
	$ sudo pip install numpy matplotlib pandas ipython[all]

##### Install TRAPpy

    $ sudo pip install --upgrade trappy

# Quickstart

Now launch the ipython notebook server:

    $ ipython notebook

This should pop up a browser. If it doesn't, open a web browser and go
to http://localhost:8888/tree/

In the `doc/` folder there's a `00 - Quick start` which describes how to
run TRAPpy. Other notebooks in that directory describe other functions
of TRAPpy.

# Documentation

API reference can be found in https://pythonhosted.org/TRAPpy/

# For developers

## Clone the repository

The code of the TRAPpy toolkit with all the supported tests and
Notebooks can be cloned from the official GitHub repository with this
command:

    $ git clone https://github.com/ARM-software/trappy.git

## Testing your installation

An easy way to test your installation is to use the `nosetests` command from
TRAPpy's home directory:

	$ nosetests

If the installation is correct all tests will succeed.
