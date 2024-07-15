.. LISA documentation master file, created by
   sphinx-quickstart on Tue Dec 13 14:20:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
  :maxdepth: 2
  :hidden:

  sections/guides/index
  sections/tools/index
  sections/api/index
  sections/changes/index

LISA Documentation
==================

Welcome to LISA documentation. LISA - "Linux Integrated System Analysis" is a
toolkit for interactive analysis and automated regression testing of Linux
kernel behaviour. LISA's goal is to help Linux kernel developers measure the
impact of modifications in core parts of the kernel. The focus is on the
scheduler (e.g. EAS), power management and thermal frameworks. However LISA is
generic and can be used for other purposes.

LISA has a "host"/"target" model. LISA itself runs on a host machine, and uses
the :mod`devlib` package to interact with the target via SSH or ADB. LISA is
flexible with regard to the target OS; its only expectation is a Linux
kernel-based system. Android, GNU/Linux and busybox style systems have all been
used.

LISA provides features to describe workloads (notably using rt-app) and run
them on targets. It can collect trace files from the target OS (e.g. ftrace
traces) and parse them. These traces can then be parsed and analysed in order
to examine detailed target behaviour during the workload's execution.

See :ref:`getting-started-page` for setup instruction.

See https://gitlab.arm.com/tooling/lisa for the source repository.
