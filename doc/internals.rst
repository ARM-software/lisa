**************
LISA internals
**************

TestEnv
=======

At the core of :class:`~lisa.env.TestEnv` is
:class:`~devlib.target.Target`. In short, it's a device
communication abstraction library that gives us a simple Python interface for
playing around with a device (shell, file transfer...). Have a look at its
documentation for more details.

As a rule of thumb, if you want to add a feature to :class:`~lisa.env.TestEnv`
that only depends on :class:`~devlib.target.Target`, chances are this should
be contributed to Devlib instead.

.. automodule:: lisa.env
   :members:

Energy model
============

.. automodule:: lisa.energy_model
   :members:

Miscellaneous utilities
=======================

.. automodule:: lisa.utils
   :members:
