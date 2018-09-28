**************
Kernel testing
**************

Introduction
============

The LISA kernel tests are mostly meant for regression testing, or for supporting
new submissions to LKML. The existing infrastructure can also be used to hack up
workloads that would allow you to poke specific areas of the kernel.

Tests do not **have** to target Arm platforms nor the task scheduler. The only
real requirement is to be able to abstract your target through
:mod:`devlib`, and from there you are free to implement tests as you see fit.

They are commonly split into two steps:
  1) Collect some data by doing work on the target
  2) Post-process the collected data

In our case, the data usually consists of
`Ftrace <https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces
that we then postprocess using :mod:`trappy`.

Writing tests
=============

Basics
++++++

Writing scheduler tests can be difficult, especially when you're
trying to make them work without relying on custom tracepoints (which is
what you should aim for). Sometimes, a good chunk of the test code will be
about trying to get the target in an expected initial state, or preventing some
undesired mechanic from barging in. That's why we rely on the freezer cgroup to
reduce the amount of noise introduced by the userspace, but it's not solving all
of the issues. As such, your tests should be designed to:

a. minimize the amount of non test-related noise (freezer, disable some
   devlib module...)
b. withstand events we can't control (use error margins, averages...)

Where to start
++++++++++++++

The main class of the kernel tests is :class:`~lisa.tests.kernel.test_bundle.TestBundle`.
Have a look at its documentation for implementation and usage examples. It can
also be worth to look at a simple test, such as
:class:`~lisa.tests.kernel.misc.capacity_sanity.CapacitySanityCheck`.

Implementations of :class:`~lisa.tests.kernel.test_bundle.TestBundle` can
execute any sort of arbitry Python code. This means that you are free to
manipulate sysfs entries, or to execute arbitray binaries on the target. The
:class:`~lisa.tests.kernel.workload.Workload` class has been created to
facilitate the execution of commands/binaries on the target.

An important daughter class of :class:`~lisa.wlgen.workload.Workload`
is :class:`~lisa.wlgen.rta.RTA`, as it facilitates the creation and
execution of `rt-app <https://github.com/scheduler-tools/rt-app>`_ workloads.
It is very useful for scheduler-related tests, as it makes it easy to create
tasks with a pre-determined utilization.

API
===

Base classes
++++++++++++

.. automodule:: lisa.tests.kernel.test_bundle
   :members:

.. TODO:: Make those imports more generic

Scheduler tests
+++++++++++++++

EAS tests
---------
.. automodule:: lisa.tests.kernel.scheduler.eas_behaviour
   :members:

Misfit tests
------------

.. automodule:: lisa.tests.kernel.scheduler.misfit
   :members:

Hotplug tests
+++++++++++++
.. automodule:: lisa.tests.kernel.hotplug.torture
   :members:

Miscellaneous tests
+++++++++++++++++++
.. automodule:: lisa.tests.kernel.misc.capacity_sanity
   :members:
