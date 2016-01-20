# Introduction

This project provides a collection of tools to support regression testing and
interactive analysis of workload behavior. Its goal is to support Linux kernel
developers to measure the impact of modifications in core parts of the kernel.
The focus is on scheduler, power management and thermal frameworks, however the
toolkit is generic to be used for other purposes.

The toolkit depends on a set of external core libraries to provide an API, and
a set of test-cases to support regression testing on core kernel features.  A
set of IPython Notebooks also allows live experiments on a target and supports
the development and testing of new use-cases.

This is an overall view of the toolkit:

	+---------------------------------------+
	| +-----------------------------------+ |
	| |           LISA Toolkit            | |
	| |  +-----------------------------+  | |
	| |  |      IPython Notebooks      |  | |
	| |  +-----------------------------+  | |
	| |  +-------+ +-------+ +---------+  | |
	| |  | wlgen | | tests | | reports |  | |
	| |  +-------+ +-------+ +---------+  | |
	| +-----------------------------------+ |
	| +----------+ +-------+                |
	| |          | | BART  |                |
	| |          | +-------+                |
	| |  devlib  | +----------------------+ |
	| |          | |      TRAPpy          | |
	| +--^-------+ +----------------------+ |
	|   ||                                  |
	|   ||                           HOST   |
	+---------------------------------------+
	    ||
	    || SSH/ADB
	    |+
	+---V-----------------------------------+
	|                              TARGET   |
	|                                       |
	|                         Linux/Android |
	|                          or Localhost |
	+---------------------------------------+


The core python libraries provide documentation of their APIs:

- [devlib](http://github.com/ARM-software/devlib)
  Documentation: [online](https://pythonhosted.org/devlib/index.html)
- [TRAPpy](http://github.com/ARM-software/trappy)
  Documentation: [online](http://arm-software.github.io/trappy/)
- [BART](http://github.com/ARM-software/bart)
  Documentation: [online](http://arm-software.github.io/bart/)


# Installation

This notes assumes an installation from scratch on a freshly installed Debian
system.

## Required dependencies

##### Install build essential tools

	$ sudo apt-get install build-essential autoconf automake libtool pkg-config

##### Install additiona tools required for some tests and functionalities

	$ sudo apt-get install nmap trace-cmd

##### Install required python packages

	$ sudo apt-get install python-matplotlib python-numpy libfreetype6-dev libpng12-dev

##### Install the Python package manager

	$ sudo apt-get install python-pip python-dev

##### Install (upgrade) required Python libraries

	$ sudo pip install --upgrade trappy bart-py devlib

## Clone the repository

TODO: add notes on how to clone and initialize the repostiory


# Target platform requirements

The target to use for the experiments must satisfy these requirements:

## Linux Targets

- allow *ssh* access, preferably as root, using either a password or an SSH key
- support *sudo*, even if it's accessed as root user

## Android Targets

- allow *adb* access, eventually by specifying a DEVICE ID
- the local shell should define the *ANDROID_HOME* environment variable pointing
  to an Android SDK installation

## Kernel features

Most of the tests targets a kernel with support for some new frameworks which
are currently in-development:

- Energy-Aware Scheduler (EAS)
- SchedFreq: the CPUFreq governor
- SchedTune: the central, scheduler-driven, power-perfomance control

Tests targeting an evaluation of these frameworks requires also a set of
tracepoint which are not available in mainline kernel. The series of patches
required to add to a recent kernel the tracepoints required by some tests are
available on this git repository:

  git://www.linux-arm.org/linux-power.git lisa/debug

The patches required are: lisa/debug_base..lisa/debug


# Toolkit organization

The toolkit provides these resources:

A collection of "assets" required to run examples and tests:
	.
	|-- assets
	|   `-- mp3-short.json

A script to setup the test environment.

	|-- init_env

A collection of IPython Notebooks grouped by topic:

	|-- ipynb
	|   |-- devlib
	|   |-- ipyserver_start
	|   |-- ipyserver_stop
	|   |-- nohup.out
	|   |-- sched_dvfs
	|   |-- utils
	|   `-- wlgen

A set of support libraries to easily develop new IPython notebooks and tests:

	|-- libs
	|   |-- __init__.py
	|   |-- utils
	|   `-- wlgen

A JSON based configuration for the target device to use for running the tests:

	|-- target.config

A collection of regression tests grouped by topic:

	|-- tests
	|   `-- eas

A collection of binary tools, pre-compiled for different architectures, which
are required to run tests on the target device:

	`-- tools
	    |-- arm64
	    |-- armeabi
	    |-- plots.py
	    |-- report.py
	    |-- scripts
	    `-- x86

(these are provided in binary form because they correspond to specific
known-good versions and it avoids having to cross-compile for the target)

# Quickstart tutorial

This section provide a quick start guide on understanding the toolkit by guiding
the user though a set of example usages.


## 1. IPython server

[IPython](http://ipython.org/notebook.html) is a web based interactive python
programming interface. This toolkit provides a set of IPython notebooks ready
to use. To use these notebooks, an IPython server must be started to serve up
html plots to a browser on the same machine or over a local network.

	# Enter the ipynb folder
	$ cd ipynb
	# Start the server
        $ ./ipyserver_start lo

This will start the server and open the index page in a new browser tab.  If
the index is not automatically loaded in the browser, visit the link reported
by the server startup script.

The index page is an HTML representation of the local ipynb folder.
From that page we can access all the IPython notebooks provided by the toolkit.


## 2. Setup the TestEnv module

Typical notebooks and tests will make use of the TestEnv class to initialize
and access a remote target device. An overall view of the functionalities
exposed by this class can be seen in this notebook:
[utils/testenv_example.ipynb](http://localhost:8888/notebooks/utils/testenv_example.ipynb)


## 3. Typical experiment workflow

RT-App is a configurable synthetic workload generator used to run different
intensity experiments on a target. The toolkit provides a python API to
simplify the definition of RT-App based workloads and their execution on a
target.

This notebook:
[wlgen/simple_rtapp.ipynb](http://localhost:8888/notebooks/wlgen/simple_rtapp.ipynb)
is a complete example of experiment setup, execution and data collection.

Specifically it shows how to:
1. configure a target for an experiment
2. configure FTrace for event collection
3. configure an HWMon based energy meter for energy measurements
4. configure a simple rt-app based workload consisting of two different tasks
5. run the workload on the target to collect FTrace events and energy
   consumption
6. visualize scheduling events using the in-browser trace plotter provided by
   the TRAPpy library
7. visualize some simple performance metrics for the tasks


## 4. Example of a smoke test for sched-DVFS

One of the main aims of this toolkit is to become a repository for
regression tests on scheduler and power management behavior.
A common pattern on defining new test cases is to start from an IPython
notebook to design an experiment and compute metrics of interest.
The notebook can then be converted into a self-contained test to run in batch
mode.

An example of such a notebook is:
[sched_dvfs/smoke_test.ipynb](http://localhost:8888/notebooks/sched_dvfs/smoke_test.ipynb)

In this notebook the toolkit API is more extensively used to defined an
experiment to:
1. select and configure three different CPUFreq governor
2. run a couple of RTApp based test workloads in each configuration
3. collect and plot scheduler and CPUFreq events
4. collect and compare the energy consumption during workload execution
   in each of the different configurations

The notebook compares three different CPUFreq governor: "performance", "sched"
and "ondemand". New configurations are easy to add. For each configuration the
notebook generate plots and tabular reports regarding working frequencies and
energy consumption.

This notebook is a good example of usage of the toolkit to define a new set of
experiments which can than be transformed into a standalone regression test.

## 5. Example of regression test: EAS RFC

Once a new set of tests have been defined and verified, perhaps using a
notebook to develop them, they can be transformed into a standalone regression
test. Regression tests are written using the same API used to develop a
notebook, thus their transformation in a batch task is generally quite easy,
especially considering that a notebook can be exported as a standalone python
script.

An example of such a regression test is:

	tests/eas/rfc.py

This test is designed to allow different configurations and workloads to be
compared from both a performance and an energy standpoint.

To run this regression test, first set up the local execution environment by
sourcing the initialization script:

	$ source init_env

Next, check the target configuration which is defined in *target.config*. This
file has to be updated to at least define the login credentials for the target
to use, and the kind of platform (e.g. "linux" or "android") and the board (if
it is supported by the toolkit, e.g. "tc2" or "juno")

The set of target configuration considered by the test as well as the set of
workloads to execute on each configuration is defined by a test specific
configuration file. In the case of the EAS regression suite, this file is
*tests/eas/rfc_eas.config*.  Have a look at this file and ensure to enable/tune
the "confs" and "wloads" sections. The default configuration runs a predefined
set of tests which are commonly used for EAS RFC postings.

Once eveything has been configured to run the test, execute it with:

	nosetests -v tests/eas/rfc.py:EAS

This command will run all the configured experiments and collect the results
into the output folder generated by the TestEnv and pointed to by
"results_latest" symlink in the top folder.

Once the test has completed, report the results using the command:

	./tools/report.py  --base noeas --tests eas

which will generate a table comparing energy/performance metrics for the "eas"
configuration with respect to the "noeas" configuration.

### Target configuration

Regression tests make use of the test environment generated by the TestEnv
module. By default, this module configures the target defined by the
__target.conf__ file present at the top level folder.

The comments in this file should be good enought to understand how to properly
setup a target to be used for the execution of all the tests provided by the
toolkit.

### Experiments configuration

The configuration of a specific regression test is usually provided by a
corresponding configuration file. For example, the configuration file for the
tests/eas/rfc.py tests is provided by the __tests/eas/rfc.conf__.

This configuration file describes:
1. which devlib modules are required by this experiment
2. which binary tools needs to be deployed in the target to run the experiments
3. other devlib specific configurations (e.g. FTrace events of interest)
4. the set of __target configurations__ (confs) to test
5. the set of __workloads__ (wloads) to run

The test will run each workload with each specified target kernel
configuration.

### Results reporting

The results of a specific experiment execution can be obtained once the test
has completed using this command:

	./tools/report.py --bases <regexp1> --tests <regexp2>

This script compares a base configuration (which name matches the
regular expression __regexp1__), to each target configuration which
name matches the __regexp2__ regular expression.

### Plotting data

A default set of plots can be generated for all the executed experiments using
this command:

  	./tools/plot.py

This script will produce, for each run of the experiments, a set of plots saved
as PNG images into the output folder of the experiment.


# Contributions / Pull Requests

All contributions are Apache 2.0. Only submit contributions where you have
authored all of the code. If you do this on work time make sure your employer
is cool with this.

