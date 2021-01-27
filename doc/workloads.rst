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
class, see the documentation for common functionalities.

rt-app
++++++

Sysbench
++++++++

Sysbench is a useful workload to get some performance numbers, e.g. to
assert that higher frequencies lead to more work done
(as done in :class:`~lisa.tests.cpufreq.sanity.UserspaceSanity`).

See :class:`~lisa.wlgen.sysbench.Sysbench` for more details.

Target script
+++++++++++++

The :class:`~lisa.target_script.TargetScript` class lets you define a Bash
script to be later executed on the target.

It is useful when you want generate some activity on the target without the
overhead of an ssh/adb connection.

API
===

Base class
++++++++++

.. automodule:: lisa.wlgen.workload
   :members:

rt-app
++++++

.. automodule:: lisa.wlgen.rta
   :members:

Sysbench
++++++++

.. automodule:: lisa.wlgen.sysbench
   :members:

Target script
+++++++++++++

.. automodule:: lisa.target_script
   :members:
