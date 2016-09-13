BART [![Build Status](https://travis-ci.org/ARM-software/bart.svg?branch=master)](https://travis-ci.org/ARM-software/bart) [![Version](https://img.shields.io/pypi/v/bart-py.svg)](https://pypi.python.org/pypi/bart-py)
====

The Behavioural Analysis and Regression Toolkit is based on
[TRAPpy](https://github.com/ARM-software/trappy). The primary goal is to assert
behaviours using the FTrace output from the kernel.

## Target Audience

The framework is designed to cater to a wide range of audience. Aiding
developers as well as automating the testing of "difficult to test" behaviours.

#### Kernel Developers

Making sure that the code that you are writing is doing the right thing.

#### Performance Engineers

Plotting/Asserting performance behaviours between different revisions of the
kernel.

#### Quality Assurance/Release Engineers

Verifying behaviours when different components/patches are integrated.

# Installation

The following instructions are for Ubuntu 14.04 LTS but they should
also work with Debian jessie.  Older versions of Ubuntu or Debian
(e.g. Ubuntu 12.04 or Debian wheezy) will likely require to install
more packages from pip as the ones present in Ubuntu 12.04 or Debian
wheezy will probably be too old.

## Required dependencies

#### Install additional tools required for some tests and functionalities

    $ sudo apt install trace-cmd kernelshark

#### Install the Python package manager

    $ sudo apt install python-pip python-dev

#### Install required python packages

    $ sudo apt install libfreetype6-dev libpng12-dev python-nose
    $ sudo pip install numpy matplotlib pandas ipython[all]
    $ sudo pip install --upgrade trappy

`ipython[all]` will install [IPython
Notebook](http://ipython.org/notebook.html), a web based interactive
python programming interface.  It is required if you plan to use interactive
plotting in BART.

#### Install BART

    $ sudo pip install --upgrade bart-py

# For developers

Instead of installing TRAPpy and BART using `pip` you should clone the repositories:

    $ git clone git@github.com:ARM-software/bart.git
    $ git clone git@github.com:ARM-software/trappy.git

Add the directories to your PYTHONPATH

    $ export PYTHONPATH=$BASE_DIR/bart:$BASE_DIR/trappy:$PYTHONPATH


# Trace Analysis Language

BART also provides a generic Trace Analysis Language, which allows the user to
construct complex relation statements on trace data and assert their expected
behaviours. The usage of the Analyzer module can be seen for the thermal
behaviours
[here](https://github.com/ARM-software/bart/blob/master/docs/notebooks/thermal/Thermal.ipynb)

# Scheduler Assertions

Enables assertion and the calculation of the following parameters:

#### Runtime

The total time that the task spent on a CPU executing.

#### Switch

Assert that a task switched between CPUs/Clusters in a given window of time.

#### Duty Cycle

The ratio of the execution time to the total time.

#### Period

The average difference between two switch-in or two switch-out events of a
task.

#### First CPU

The first CPU that a task ran on.

#### Residency

Calculate and assert the total residency of a task on a CPU or cluster.

#### Examples

The Scheduler assertions also use TRAPpy's EventPlot to provide a `kernelshark`
like timeline for the tasks under consideration. (in IPython notebooks).

A notebook explaining the usage of the framework for asserting the deadline
scheduler behaviours can be seen
[here](https://rawgit.com/sinkap/0abbcc4918eb228b8887/raw/a1b4d6e0079f4ea0368d595d335bc340616501ff/SchedDeadline.html).

# API reference

The API reference can be found in https://pythonhosted.org/bart-py
