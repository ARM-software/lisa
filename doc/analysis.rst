*********************
Kernel trace analysis
*********************

Base class
==========

.. automodule:: lisa.analysis.base
   :members:

Load tracking
=============

.. automodule:: lisa.analysis.load_tracking
   :members:

CPUs
====

.. automodule:: lisa.analysis.cpus
   :members:

Frequency
=========

.. automodule:: lisa.analysis.frequency
   :members:

Tasks
=====

.. These two autoclasses should not be necessary, but sphinx doesn't seem
   to like Enums and refuses to do anything with TaskState unless explicetely
   told to.

.. autoclass:: lisa.analysis.tasks.StateInt
   :members:

.. autoclass:: lisa.analysis.tasks.TaskState
   :members:

.. automodule:: lisa.analysis.tasks
   :members:
   :exclude-members: StateInt, TaskState

Idle
====

.. automodule:: lisa.analysis.idle
   :members:

Latency
=======

.. automodule:: lisa.analysis.latency
   :members:
