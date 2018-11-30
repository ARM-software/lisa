**************
LISA internals
**************

Target
======

At the core of :class:`~lisa.target.Target` is
:class:`~devlib.target.Target`. In short, it's a device
communication abstraction library that gives us a simple Python interface for
playing around with a device (shell, file transfer...). Have a look at its
documentation for more details.

As a rule of thumb, if you want to add a feature to
:class:`~lisa.target.Target`, chances are this should be contributed to Devlib
instead.

.. automodule:: lisa.target
   :members:

Configuration management
========================

Configuration files are managed by sublcasses of
:class:`lisa.conf.MultiSrcConf`. It allows loading from a YAML file (not to be
confused with serializing the instance).


.. automodule:: lisa.conf
   :members:

PlatformInfo
============

The main source of information for tests come from :class:`~lisa.trace.Trace`
and :class:`~lisa.platforms.platinfo.PlatformInfo`. The latter gives access to
information autodetected from the :class:`devlib.target.Target` or filled in by
the user.

.. automodule:: lisa.platforms.platinfo
   :members:

Energy model
============

.. automodule:: lisa.energy_model
   :members:
   :private-members: _CpuTree

Trace
=====

.. automodule:: lisa.trace
   :members:

Performance Analysis
====================

.. automodule:: lisa.perf_analysis
   :members:

Miscellaneous utilities
=======================

.. automodule:: lisa.utils
   :members:

.. automodule:: lisa.target_script
   :members:

.. automodule:: lisa.regression
   :members:
