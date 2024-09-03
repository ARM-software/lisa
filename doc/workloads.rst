*********
Workloads
*********

Introduction
============

LISA is not intended to automate the execution of complex workloads such as
Android benchmarks (Jankbench, PCMark...). For this kind of application, you
should turn to `Workload Automation <https://github.com/ARM-software/workload-automation>`_
instead.

However, a minimal amount of support for running experiments is still useful.
That is why LISA provides the wlgen API, which lets you execute commands and
binaries while providing some help for managing tasksets, cgroups, and more.

Available workloads
===================

Most of these workloads are based on the :class:`~lisa.wlgen.workload.Workload`
class, see the documentation for common functionalities:

  * Base infrastructure: :mod:`lisa.wlgen.workload`
  * ``rt-app``: :mod:`lisa.wlgen.rta`
  * ``sysbench``: :mod:`lisa.wlgen.sysbench`
