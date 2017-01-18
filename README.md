
*__NOTE__: This is still a work in progress project, suitable for:*
*developers, contributors and testers.*
*None of the provided tests should be considered stable and/or suitable*
*for the evaluation of a product.*

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

# Documentation

* [Wiki Home page](https://github.com/ARM-software/lisa/wiki)
* [Installation](https://github.com/ARM-software/lisa/wiki/Installation)
* [Quickstart Tutorial](https://github.com/ARM-software/lisa/wiki/Quickstart-tutorial)

# License

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source licenses.  For more information, see lisa/tools/LICENSE.*

# Contributions / Pull Requests

Contributions are accepted under Apache-2.0. Only submit contributions where you have
authored all of the code. If you do this on work time make sure your employer
is cool with this.
