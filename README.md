
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

# Installation

This note assumes installation from scratch on a freshly installed
Ubuntu 16.04 system. If you have a different system (e.g. OSX or an
older version of Ubuntu, you can probably be interesting into a virtual
machine based installation, which is described in a following section.

## Required dependencies

	# Install common build related tools
	$ sudo apt-get install build-essential autoconf automake libtool pkg-config

	# Install additional tools required for some notebooks and tests
	$ sudo apt-get install trace-cmd sshpass kernelshark

	# Install optional tools required for some notebooks and tests
	$ sudo apt-get install nmap net-tools tree

	# Install required python packages
	$ sudo apt-get install python-matplotlib python-numpy libfreetype6-dev libpng12-dev python-nose

	# Install the Python package manager
	$ sudo apt-get install python-pip python-dev

	# Install (upgrade) required Python libraries
	$ sudo pip install --upgrade trappy bart-py devlib

*NOTE:* TRAPpy and BART depend on *ipython* and *jupyter*. Some IPython
Notebooks examples are written using the notebooks JSON nbformat version 4,
which might not be supported by the IPython version installed by *apt-get*.
It is suggested to remove apt-get installed IPython and install it
using *pip*, which will provides the most updated version:

	# Remove (eventually) already installed versions
	$ sudo apt-get remove ipython ipython-notebook
	# Install most update version of the notebook
	$ sudo pip install ipython jupyter

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
available on this git repository and branch:

	git://www.linux-arm.org/linux-pb.git lisa/debug

The patches required are the ones in this series:

	git log --oneline lisa/debug_base..lisa/debug
	
# Virtual Machine based installation

LISA provides a Vagrant recipe which allows to automate the generation of a
VirtualBox based virtual machine pre-configured to run LISA. To generate
and use such a virtual machine you need:

- VirtualBox intalled in your machine, you can download the VM installer for
  your specific system from this page: https://www.virtualbox.org/wiki/Downloads

- Vagrant installed in your machine, you can download the installer for your
  specific system from this page: https://www.vagrantup.com/downloads.html
```bash
$ wget https://releases.hashicorp.com/vagrant/1.8.1/vagrant_1.8.1_x86_64.deb
$ sudo dpkg -i ./vagrant_1.8.1_x86_64.deb
```
Once these two components are available in your machine, to install LISA you
need to:

- clone the LISA repository in a local folder
```bash
# Clone the master LISA repository
$ git clone https://github.com/ARM-software/lisa.git
```
- create and start a Vagrant/VirtualBox VM
```bash
# Enter the LISA source tree
$ cd lisa
# Install LISA and its dependencies within the virtual machine
$ vagrant up
```	
This last command builds and execute the VM according to the description
provided by the Vagrant file available in the root folder of the LISA
source tree. The first time you run this command it will take some time
to download the based Ubuntu image and to install the required LISA
dependencies. The actual time depends on the speed of your internet
connection, the download size is rought 

When the installation complete you will get a prompt from the LISA shell
which is running withint the VM you just built. This VM shell can be
accessed from another terminal using this command
```bash
# from within the LISA root folder...
$ vagrant ssh
```
Once you exit all the LISA shell the VM is automatically stopped by vagrant.
The next time you run the "up" command the VM will be started again and you
will get a LISA shell.

From within the LISA shell you can start the IPython Notebooks server
by following the instructions in the following section.

# Quickstart tutorial

Once cloned, source init_env to initialized the LISA Shell, which provides
a convenient set of shell commands for easy access to many LISA related
functions.

```shell
$ source init_env
```

To start the IPython Notebook Server required to use this Notebook, on a
LISAShell run:

```shell
[LISAShell lisa] \> lisa-ipython start

Starting IPython Notebooks...
Starting IPython Notebook server...
  IP Address :  http://127.0.0.1:8888/
  Folder     :  /home/derkling/Code/lisa/ipynb
  Logfile    :  /home/derkling/Code/lisa/ipynb/server.log
  PYTHONPATH : 
    /home/derkling/Code/lisa/libs/bart
    /home/derkling/Code/lisa/libs/trappy
    /home/derkling/Code/lisa/libs/devlib
    /home/derkling/Code/lisa/libs/wlgen
    /home/derkling/Code/lisa/libs/utils


Notebook server task: [1] 24745
```

The main folder served by the server is:
  http://127.0.0.1:8888/
  
Once the server is started you can have a look at the provide tutorial notebooks
are accessible by following (in your browser) this link:

  http://127.0.0.1:8888/notebooks/tutorial/00_LisaInANutshell.ipynp

This intial tutorial can be seen (but not executed) also on GitHub:

  https://github.com/ARM-software/lisa/blob/master/ipynb/tutorial/00_LisaInANutshell.ipynb

# License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses.  For more information, see lisa/tools/LICENSE.*

# Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have
authored all of the code. If you do this on work time make sure your employer
is cool with this.

