
*__NOTE__: This is still a work in progress project, suitable for:*
*developers, contributors and testers.*<br>
*None of the provided tests should be considered stable and/or suitable*
*for the evaluation of a product.*

# Introduction

The LISA project provides a toolkit that supports regression testing and
interactive analysis of workload behavior. LISA stands for Linux
Integrated/Interactive System Analysis. LISA's goal is to help Linux
kernel developers to measure the impact of modifications in core parts
of the kernel.  The focus is on the scheduler, power management and
thermal frameworks. However LISA is generic and can be used for other
purposes too.

LISA provides an API for modeling use-cases of interest and developing
regression tests for use-cases.  A ready made set of test-cases to
support regression testing of core kernel features is provided.  In
addition, LISA uses the excellent IPython notebook framework and a set
of notebooks are provided for live experiments on a target platform.

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

LISA depends on the following external python libraries:

- [devlib](http://github.com/ARM-software/devlib)
  Documentation: [online](https://pythonhosted.org/devlib/index.html)
- [TRAPpy](http://github.com/ARM-software/trappy)
  Documentation: [online](http://arm-software.github.io/trappy/)
- [BART](http://github.com/ARM-software/bart)
  Documentation: [online](http://arm-software.github.io/bart/)

# Installation

This note assumes installation from scratch on a freshly installed
Debian system.

## Required dependencies

##### Install common build related tools

	$ sudo apt-get install build-essential autoconf automake libtool pkg-config

##### Install additional tools required for some tests and functionalities

	$ sudo apt-get install nmap trace-cmd sshpass kernelshark net-tools

##### Install required python packages

	$ sudo apt-get install python-matplotlib python-numpy libfreetype6-dev libpng12-dev

##### Install the Python package manager

	$ sudo apt-get install python-pip python-dev

##### Install (upgrade) required Python libraries

	$ sudo pip install --upgrade trappy bart-py devlib

*NOTE:* TRAPpy and BART depend on *ipython* and *ipython-notebook*. Some IPython
Notebooks examples are written in JSON nbformat version 4 which might not be
supported by the IPython version installed by *apt-get* (current version is
1.2.1-2 which does not support such format). In this case, it is needed to
remove IPython and install it using *pip* instead:

	$ sudo apt-get remove ipython ipython-notebook
	$ sudo pip install ipython ipython-notebook

## Clone the repository

The code of the LISA toolkit with all the supported tests and Notebooks can be
cloned from the official GitHub repository with this command:

        $ git clone https://github.com/ARM-software/lisa.git

# Target platform requirements

The target platform to be used for experiments with LISA must satisfy
the following requirements:

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

# Testing your installation

An easy way to test your installation is to give a run to the EAS RFC tests.
These are a set of experiments which allows to compare EAS performance and
energy consumption with respect to standard kernel.

*NOTE:* The following
[tutorial](https://github.com/ARM-software/lisa#quickstart-tutorial) is still
recommended it you want to get a better grasp on how the framework is organized
and how to use it at your best.

Let's assume your target is running an EAS enabled kernel, to run such tests
just run these few steps:

1. Check the *target.config* file in the root folder of the toolkit<br>
   In that file you to specify the proper value for at least *platform* and
   *board*, as well as the *host* IP address of the target and the login
   credentials (i.e. *username* and *password*).

2. Setup the execution environment by sourcing the provided initialization file
   from the shell you will use to run the experiments

	```sh
	$ source init_env

	```

3. Run the EAS RFC test using the standard nosetest command:

	```sh
	$ nosetests -v tests/eas/rfc.py:EAS
	```

4. Wait for the test to complete and than you can report the results with:

	```sh
	./tools/report.py  --base noeas --tests eas
	```

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
known-good versions and it avoids having to cross-compile these tools
for the target)

# Quickstart tutorial

This section provides a quick start guide on understanding the toolkit
by guiding the user though a set of example usage scenarios.

## 1. IPython server

[IPython](http://ipython.org/notebook.html) is a web based interactive python
programming interface. This toolkit provides a set of IPython notebooks ready
to use. To use these notebooks, an IPython server must be started to
serve pages to a browser on the same host machine or over a local
network.

	# Enter the ipynb folder
	$ cd ipynb
	# Start the server
        $ ./ipyserver_start lo

This will start the IPython server and open the index page in a new
browser tab.  If the index is not automatically loaded in the browser,
visit the link reported by the server startup script.

The index page is an HTML representation of the local ipynb folder.
From that page we can access all the IPython notebooks provided by the toolkit.

## 2. Setup the TestEnv module

Typical notebooks and tests will make use of the TestEnv class to initialize
and access a remote target device. An overall view of the functionality
exposed by this class can be seen in this notebook:
[utils/testenv_example.ipynb](http://localhost:8888/notebooks/utils/testenv_example.ipynb)

## 3. Typical experiment workflow

RT-App is a configurable synthetic workload generator used to run different
intensity experiments on a target. LISA provides a python wrapper API to
simplify the definition of RT-App based workloads and their execution on a
target.

This notebook:
[wlgen/simple_rtapp.ipynb](http://localhost:8888/notebooks/wlgen/simple_rtapp.ipynb)
is a complete example of setting up an experiment, execution and data
collection.

Specifically it demonstrates how to:

1. configure a target for an experiment
2. configure FTrace for event collection
3. configure an HWMon based energy meter for energy measurements
4. configure a simple rt-app based workload consisting of two different tasks
5. run the workload on the target to collect FTrace events and energy
   consumption
6. visualize scheduling events using the in-browser trace plotter provided by
   the TRAPpy library
7. visualize some simple performance metrics for the tasks

## 4. Regression test example 1

One of the main aims of LISA is to become a repository for regression
tests on scheduler and power management behavior. A common pattern for
defining new test cases is to start with an IPython notebook to design
an experiment and compute metrics of interest. The notebook can then be
converted into a self-contained test to run in batch mode.

An example of such a notebook is:
[sched_dvfs/smoke_test.ipynb](http://localhost:8888/notebooks/sched_dvfs/smoke_test.ipynb)

sched-DVFS is a technique for scheduler driven DVFS operation and is a
key part of Energy Aware Scheduling (EAS). LISA is used extensively for
EAS analysis and some of the examples listed in this README are taken
from the EAS testing and evaluation experience. To know more about EAS
and sched-DVFS, see:
(http://www.linaro.org/blog/core-dump/energy-aware-scheduling-eas-progress-update/)

In this notebook the toolkit API is more extensively used to define an
experiment to:

1. select and configure three different CPUFreq governors
2. run a couple of RTApp based test workloads in each configuration
3. collect and plot scheduler and CPUFreq events
4. collect and compare the energy consumption during workload execution
   in each of the different configurations

The notebook compares three different CPUFreq governors: "performance",
"sched" and "ondemand". New configurations are easy to add. For each
configuration the notebook generates plots and tabular reports regarding
working frequencies and energy consumption.

This notebook is a good example of using LISA to build a new set of
experiments which can then be transformed into a standalone regression
test.

## 5. Regression test example 2

Once a new set of tests have been defined and verified, perhaps by using
a notebook to develop them, they can be transformed into a standalone
regression test. Regression tests are written using the same API used to
develop a notebook, thus their transformation into a batch task is
generally quite easy, especially considering that a notebook can be
exported as a standalone python script.

An example of such a regression test is:

	tests/eas/rfc.py

This test, which is used for EAS analysis, is designed to allow
different configurations and workloads to be compared from both a
performance and an energy standpoint.

To run this regression test, first set up the local execution environment by
sourcing the initialization script:

	$ source init_env

Next, check the target configuration which is defined in *target.config*. This
file has to be updated to at least define the login credentials for the target
to use, and the kind of platform (e.g. "linux" or "android") and the board (if
it is supported by the toolkit, e.g. "tc2" or "juno")

The set of target configurations considered by the test as well as the
set of workloads to execute with each configuration is defined by a test
specific configuration file. In the case of the EAS regression suite,
this file is *tests/eas/rfc_eas.config*. Have a look at this file and
ensure to enable/tune the "confs" and "wloads" sections. The default
configuration runs a predefined set of tests which are commonly used for
EAS RFC postings.

Once eveything has been configured to run the test, execute it with:

	nosetests -v tests/eas/rfc.py:EAS

This command will run all the configured experiments and collect the results
into the output folder generated by the TestEnv and pointed to by the
"results_latest" symlink in the top folder.

Once the test has completed, report the results using the command:

	./tools/report.py  --base noeas --tests eas

This will generate a table comparing energy/performance metrics for the
"eas" configuration with respect to the "noeas" configuration.

### Target configuration

Regression tests make use of the test environment generated by the TestEnv
module. By default, this module configures the target defined by the
__target.conf__ file present at the top level folder.

The comments in this file should be good enough to understand how to
properly setup a target to be used for the execution of all the tests
provided by the toolkit.

### Experiments configuration

The configuration of a specific regression test is usually provided by a
corresponding configuration file. For example, the configuration file for the
tests/eas/rfc.py tests is provided by the __tests/eas/rfc.conf__.

This configuration file describes:

1. which devlib modules are required by this experiment
2. which binary tools need to be deployed in the target to run the
   experiments
3. other devlib specific configurations (e.g. FTrace events of interest)
4. the set of __target configurations__ (confs) to test
5. the set of __workloads__ (wloads) to run

The test will run each workload with each specified target kernel
configuration.

### Results reporting

The results of a specific experiment execution can be obtained once the test
has completed using this command:

	./tools/report.py --bases <regexp1> --tests <regexp2>

This script compares a base configuration (whose name matches the
regular expression __regexp1__), to each target configuration which
name matches the __regexp2__ regular expression.

### Plotting data

A default set of plots can be generated for all the executed experiments using
this command:

  	./tools/plot.py

This script will produce, for each run of the experiments, a set of plots saved
as PNG images into the output folder of the experiment.

# License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses.  For more information, see lisa/tools/LICENSE.*

# Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have
authored all of the code. If you do this on work time make sure your employer
is cool with this.

