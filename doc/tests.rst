**********
LISA tests
**********

Introduction
============

These tests were developped to verify patches supporting Arm's big.LITTLE
in the Linux scheduler. You can see these test results being published
`here <https://developer.arm.com/open-source/energy-aware-scheduling/eas-mainline-development>`_.

Tests do not **have** to target Arm platforms nor the scheduler. The only real
requirement is to have a :class:`libs.devlib` target handle, and from there you
are free to implement tests as you see fit.

They are commonly split into two steps:
  1) Collect some data by doing some work on the target
  2) Post-process the collected data

In our case, the data usually consists of
`Ftrace <https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces
that we then postprocess using :mod:`libs.trappy`.

Writing tests
=============

Writing scheduler tests can be difficult, especially when you're
trying to make them work without relying on custom tracepoints (which is
what you should aim for). Sometimes, a good chunk of the test code will be
about trying to get the target in an expected initial state, or preventing some
undesired mechanic from barging in. That's why we rely on the freezer cgroup to
reduce the amount of noise introduced by the userspace, but it's not solving all
of the issues. As such, your tests should be designed to:

a. minimize the amount of non test-related noise (freezer, disable some module...)
b. withstand events we can't control (use error margins, averages...)

Having tunable margins (such as for
:meth:`libs.utils.generic.GenericTestBundle.test_slack`) is also desirable, as
it allows things like parameter sweep in CI.

API
===

Base classes
------------

.. automodule:: libs.utils.test_workload
   :members:

Generic tests
-------------

.. autoclass:: libs.utils.generic.GenericTestBundle
   :members:

Hotplug tests
-------------
.. automodule:: libs.utils.cpu_hotplug
   :members:

Implemented tests
=================

Generics
--------

.. automodule:: libs.utils.generic
   :exclude-members: GenericTestBundle
   :members:
