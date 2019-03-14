******
Target
******

Introduction
============

Our :class:`~lisa.target.Target` is a wrapper around
:class:`devlib.target.Target`. In short, it's a device communication
abstraction library that gives us a simple Python interface for playing around
with a device (shell, file transfer...).

If you want to execute some command on a target, it's as simple as this::

  out = target.execute("ls -al .")

Have a look at the devlib documentation for more details. Our wrapper brings
additionnal features which are documented below.

As a rule of thumb, if you want to add a feature to :class:`~lisa.target.Target`,
chances are this should be contributed to devlib instead.

Connecting to a target
======================

Connecting to a target means creating a :class:`~lisa.target.Target` instance.
This can be as a simple as this::

  target = Target(kind="linux", host="192.168.0.1", username="root", password="root")

For more convenience, you can also save the relevant connection information for
a given target in a configuration file, which would let you create a
:class:`~lisa.target.Target` like so::

   target = Target.from_one_conf("path/to/my/conf.yml")


.. seealso::

   See the documentation of :class:`~lisa.target.Target` and
   :class:`~lisa.target.TargetConf` more details.

Platform data
=============

The main source of information for tests come from :class:`~lisa.trace.Trace`
and :class:`~lisa.platforms.platinfo.PlatformInfo`. The latter gives access to
information autodetected from the :class:`devlib.target.Target` or filled in by
the user.

This information ranges from the current kernel version to the available
platform frequencies.

.. seealso::

   See the documentation of :class:`~lisa.platforms.platinfo.PlatformInfo`
   for more details.

API
===

Target
++++++

.. automodule:: lisa.target
   :members:

Platform Info
+++++++++++++

.. automodule:: lisa.platforms.platinfo
   :members:
