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

Base API
========

.. automodule:: lisa.wlgen.workload
   :members:

Implemented workloads
=====================

rt-app
++++++

Base class
----------

.. autoclass:: lisa.wlgen.rta.RTA
   :members:

rt-app profile classes
----------------------

To make rt-app workload description easier, rt-app tasks can be described using
the :meth:`RTA.by_profile` class method. This method is powered by :class:`RTATask`
and its siblings.

.. autoclass:: lisa.wlgen.rta.Phase
   :members:

.. autoclass:: lisa.wlgen.rta.RTATask
   :members:

.. automodule:: lisa.wlgen.rta
   :exclude-members: RTA, RTATask, Phase
   :members:

sysbench
++++++++

.. automodule:: lisa.wlgen.sysbench
   :members:
