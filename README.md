
*__NOTE__: This is still a work in progress project, suitable for:*
*developers, contributors and testers.*
*None of the provided tests have been extensively evaluated as of January 2017.*

# Introduction

The LISA project provides a toolkit that supports regression testing and
interactive analysis of Linux kernel behavior. LISA stands for Linux
Integrated/Interactive System Analysis. LISA's goal is to help Linux
kernel developers to measure the impact of modifications in core parts
of the kernel.  The focus is on the scheduler (e.g. EAS), power management and
thermal frameworks. However LISA is generic and can be used for other purposes
too.

LISA has a "host"/"target" model. LISA itself runs on a *host* machine, and uses
the [devlib](https://github.com/ARM-software/lisa) toolkit to interact with the
*target* via SSH, ADB or telnet. LISA is flexible with regard to the target OS;
its only expectation is a Linux kernel-based system. Android, GNU/Linux and
busybox style systems have all been used.

LISA provides features to describe workloads (notably using
[rt-app](https://github.com/scheduler-tools/rt-app)) and run them on targets. It
can collect trace files from the target OS (e.g. systrace and ftrace traces),
parse them via the [TRAPpy](https://github.com/ARM-software/trappy)
framework. These traces can then be parsed and analysed in order to examine
detailed target behaviour during the workload's execution.

Some LISA features may require modifying the target OS. For example, in order to
collect ftrace files the target kernel must have CONFIG_DYNAMIC_FTRACE enabled.

There are two "entry points" for running LISA:

* Via the [Jupyter/IPython notebook framework](http://jupyter.org/). This allows
  LISA to be used interactively and supports visualisation of trace data. Some
  notebooks are provided with example and ready-made LISA use-cases.

* Via the automated test framework. This framework allows the development of
  automated pass/fail regression tests for kernel behaviour. The
  [BART](https://github.com/ARM-software/trappy) toolkit provides additional
  domain-specific test assertions for this use-case. LISA provides some
  ready-made automated tests under the `tests/` directory.

# Motivations

The main goals of LISA are:

* Support study of existing behaviours (i.e. *"how does PELT work?"*)
* Support analysis of new code being developed (i.e. *"what is the impact on
  existing code?"*)
* Get insights on what's not working and possibly chase down why
* Share reproducible experiments by means of a **common language** that:
    * is **flexible enough** to reproduce the same experiment on different
      targets
    * **simplifies** generation and execution of well defined workloads
    * **defines** a set of metrics to evaluate kernel behaviours
    * **enables** kernel developers to easily post process data to produce
      statistics and plots

# Documentation

* [Wiki Home page](https://github.com/ARM-software/lisa/wiki)
* [Installation](https://github.com/ARM-software/lisa/wiki/Installation)
* [Quickstart Tutorial](https://github.com/ARM-software/lisa/wiki/Quickstart-tutorial)

More formal API documentation for LISA is a work in progress, however much of
the API is currently described in the provided tutorial Jupyter notebooks.

# External Links
* Linux Integrated System Analysis (LISA) & Friends
  [Slides](http://events.linuxfoundation.org/sites/events/files/slides/ELC16_LISA_20160326.pdf)
  and [Video](https://www.youtube.com/watch?v=yXZzzUEngiU)

# License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses.  For more information, see lisa/tools/LICENSE.*

# Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have
authored all of the code. If you do this on work time make sure your employer
is cool with this.
