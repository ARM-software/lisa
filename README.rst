⚠️ LISA has moved from GitHub to GitLab and is now available at:
https://gitlab.arm.com/tooling/lisa. Please update your clone URLs and note
that the GitHub repository will not be used for any pull requests or issue
management. ⚠️

⚠️ ``lisa_tests`` package will be moved to another repository in
January 2024. When more details are available, this pull request will be
updated: https://gitlab.arm.com/tooling/lisa/-/merge_requests/2140/ ⚠️

Introduction |CI status|
===================================================

The LISA project provides a toolkit that supports regression testing and
interactive analysis of Linux kernel behavior. LISA stands for Linux
Integrated/Interactive System Analysis. LISA's goal is to help Linux
kernel developers to measure the impact of modifications in core parts
of the kernel. The focus is on the scheduler (e.g. EAS), power
management and thermal frameworks. However LISA is generic and can be
used for other purposes too.

LISA has a *host*/*target* model. LISA itself runs on a *host* machine,
and uses the `devlib <https://github.com/ARM-software/devlib>`__ toolkit
to interact with the *target* via SSH, ADB or telnet. LISA is flexible
with regard to the target OS; its only expectation is a Linux
kernel-based system. Android, GNU/Linux and busybox style systems have
all been used.

LISA provides features to describe workloads (notably using `rt-app
<https://github.com/scheduler-tools/rt-app>`__) and run them on targets. It can
collect trace files from the target OS (e.g. systrace and ftrace traces). These
traces can then be parsed and analysed in order to examine detailed target
behaviour during the workload's execution.

Some LISA features may require modifying the target OS. For example, in
order to collect ftrace files the target kernel must have
CONFIG_DYNAMIC_FTRACE enabled.

There are two "entry points" for running LISA:

-  Via the `Jupyter/IPython notebook framework <http://jupyter.org/>`__.
   This allows LISA to be used interactively and supports visualisation
   of trace data. Some notebooks are provided with example and
   ready-made LISA use-cases.

-  Via the automated test framework. This framework allows the
   development of automated pass/fail regression tests for kernel behaviour.
   LISA provides some ready-made automated tests under the ``lisa/tests/``
   directory.

Motivations
===========

The main goals of LISA are:

-  Support study of existing behaviours (i.e. *"how does PELT work?"*)
-  Support analysis of new code being developed (i.e. *"what is the
   impact on existing code?"*)
-  Get insights on what's not working and possibly chase down why
-  Share reproducible experiments by means of a **common language**
   that:

   -  is **flexible enough** to reproduce the same experiment on
      different targets
   -  **simplifies** generation and execution of well defined workloads
   -  **defines** a set of metrics to evaluate kernel behaviours
   -  **enables** kernel developers to easily post process data to
      produce statistics and plots

Documentation
=============

You should find everything on https://tooling.sites.arm.com/lisa/latest/
Here are some noteworthy sections:

   * `Installation <https://tooling.sites.arm.com/lisa/latest/setup.html>`__
   * `Kernel tests <https://tooling.sites.arm.com/lisa/latest/kernel_tests.html>`__

How to reach us
===============

Bug reports should be raised against the `GitLab issue tracker <https://gitlab.arm.com/tooling/lisa/-/issues>`__.

External Links
==============

-  Linux Integrated System Analysis (LISA) & Friends
   `Slides <http://events17.linuxfoundation.org/sites/events/files/slides/ELC16_LISA_20160326.pdf>`__
   and `Video <https://www.youtube.com/watch?v=zRlqwurYq5Y>`__

   ..
     video title: LAS16-TR04: Using Tracing to tune and optimize EAS English

   Note: the LISA classes referred by the slides are outdated, but all
   the other concepts and the overall architecture stays the same.

-  Some insights on what it takes to have reliable tests:
   `Video <https://www.youtube.com/watch?v=I_MZ9XS3_zc>`__

    ..
      video title: Scheduler behavioural testing

License
=======

This project is licensed under Apache-2.0.

This project includes some third-party code under other open source
licenses. For more information, see ``lisa/_assets/binaries/*/README.*``.

Contributions / Pull Requests
=============================

Contributions are accepted under Apache-2.0. Only submit contributions
where you have authored all of the code. If you do this on work time
make sure your employer is cool with this. We also have a `Contributor Guide
<https://tooling.sites.arm.com/lisa/latest/contributors_guide.html>`__

.. |CI status| image:: https://gitlab.arm.com/tooling/lisa/badges/main/pipeline.svg
   :target: https://gitlab.arm.com/tooling/lisa/-/commits/main
