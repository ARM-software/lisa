.. _kernel-testing-page:

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
that we then postprocess using :mod:`trappy`

.. seealso:: :ref:`analysis-page`

Available tests
===============

The following tests are available. They can be used as:

  * direct execution using ``lisa-test`` command (``LISA shell``) and ``exekall``
    (see :ref:`automated-testing-page`)
  * the individual classes/methods they are composed of can be used in custom
    scripts/jupyter notebooks (see ipynb/tests/synthetics_example.ipynb)

.. include:: test_list.rst

Running tests
=============

From the CLI
++++++++++++

The shortest path to executing a test from a shell is:

1. Update the ``target_conf.yml`` file located at the root of the repository
   with the credentials to connect to the development board (see
   :class:`~lisa.target.TargetConf` keys for more information)
2. Run the following:

  .. code-block:: sh

    # To run all tests
    lisa-test
    # To list available tests
    lisa-test --list
    # To run a test matching a pattern
    lisa-test '*test_task_placement'

More advanced workflows are described at :ref:`automated-testing-page`.

From a python environment
+++++++++++++++++++++++++

See the usage example of :class:`~lisa.tests.base.TestBundle`

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

a. minimize the amount of non test-related noise (e.g. freezer)
b. withstand events we can't control (use error margins, averages...)

Where to start
++++++++++++++

The main class of the kernel tests is :class:`~lisa.tests.base.TestBundle`.
Have a look at its documentation for implementation and usage examples.

.. tip::

   A simple test implementation worth looking at is
   :class:`~lisa.tests.scheduler.sanity.CapacitySanity`.

Implementations of :class:`~lisa.tests.base.TestBundle` can
execute any sort of arbitry Python code. This means that you are free to
manipulate sysfs entries, or to execute arbitray binaries on the target. The
:class:`~lisa.wlgen.workload.Workload` class has been created to
facilitate the execution of commands/binaries on the target.

An important daughter class of :class:`~lisa.wlgen.workload.Workload`
is :class:`~lisa.wlgen.rta.RTA`, as it facilitates the creation and
execution of `rt-app <https://github.com/scheduler-tools/rt-app>`_ workloads.
It is very useful for scheduler-related tests, as it makes it easy to create
tasks with a pre-determined utilization.

API
===

Base API
++++++++

.. automodule:: lisa.tests.base
   :members:
   :private-members: _from_target

.. TODO:: Make those imports more generic

Scheduler tests
+++++++++++++++

EAS tests
---------
.. automodule:: lisa.tests.scheduler.eas_behaviour
   :members:

Load tracking tests
-------------------
.. automodule:: lisa.tests.scheduler.load_tracking
   :members:
   :private-members: _from_target

Misfit tests
------------

.. automodule:: lisa.tests.scheduler.misfit
   :members:

Sanity tests
------------

.. automodule:: lisa.tests.scheduler.sanity
   :members:

Hotplug tests
+++++++++++++
.. automodule:: lisa.tests.hotplug.torture
   :members:

Cpufreq tests
+++++++++++++

.. automodule:: lisa.tests.cpufreq.sanity
   :members:

Staged tests
+++++++++++++

Those are tests that have been merged into LISA but whose behaviour are being
actively evaluated.
