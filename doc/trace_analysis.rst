.. _analysis-page:

*********************
Kernel trace analysis
*********************

Introduction
============

LISA comes with a plethora of analysis functions based on `Ftrace
<https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces. Under the
hood, we use :mod:`trappy` to convert the trace events into
:class:`pandas.DataFrame` which are suited to handling large data sets.

Trace
=====

Our :class:`~lisa.trace.Trace` takes an Ftrace ``trace.dat`` file as input (Systrace
``trace.html`` are also accepted), and provides access to both the raw trace events,
as well as some new dataframes built from analysing and aggregating trace events.

You can create one like so::

  trace = Trace("path/to/trace.dat", events=["sched_switch", "sched_wakeup"])

Raw trace events can be accessed like this::

  trace.df_events("sched_switch")

Whereas analysis dataframes can be obtained like that::

  # trace.analysis.<analysis name>.<analysis method>
  trace.analysis.tasks.df_tasks_states()

.. seealso:: See the :class:`~lisa.trace.Trace` documentation for more details.

Available analysis
==================

Dataframes
++++++++++

The majority of these dataframes are time-indexed (and if they aren't, it will
be called out in the docstring). This makes it easy to create dataframe slices
to study specific trace windows.

.. include:: analysis_df_list.rst

Plots
+++++

When run in a notebook, these plots will be displayed automatically. By default,
they are also saved in the same directory as your ``trace.dat``

.. TODO:: Generate some sample plots using the nosetest trace

.. include:: analysis_plot_list.rst

API
===

Trace
+++++

.. automodule:: lisa.trace
   :members:

Proxy
+++++

.. automodule:: lisa.analysis.proxy
   :members:

Base class
++++++++++

.. automodule:: lisa.analysis.base
   :members:

Load tracking
+++++++++++++

.. automodule:: lisa.analysis.load_tracking
   :members:

CPUs
++++

.. automodule:: lisa.analysis.cpus
   :members:

Frequency
+++++++++

.. automodule:: lisa.analysis.frequency
   :members:

Tasks
+++++

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
++++

.. automodule:: lisa.analysis.idle
   :members:

Latency
+++++++

.. automodule:: lisa.analysis.latency
   :members:

Status
++++++

.. automodule:: lisa.analysis.status
   :members:

Thermal
+++++++

.. automodule:: lisa.analysis.thermal
   :members:

Function profiling
++++++++++++++++++

.. automodule:: lisa.analysis.functions
   :members:
