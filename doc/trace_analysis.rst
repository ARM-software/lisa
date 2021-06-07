.. _analysis-page:

*********************
Kernel trace analysis
*********************

Introduction
============

LISA comes with a plethora of analysis functions based on `Ftrace
<https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces. We
convert the trace events into :class:`pandas.DataFrame` which are suited to
handling mid-sized data sets.

Trace
=====

Our :class:`~lisa.trace.Trace` takes an Ftrace ``trace.dat`` file as input
(Systrace ``trace.html`` are also accepted, but mileage may vary since it's an
intrinsically ambiguous format), and provides access to both the raw trace
events, as well as some new :class:`pandas.DataFrame` built from analysing and
aggregating trace events.

You can create one like so::

  trace = Trace("path/to/trace.dat")

Raw trace events can be accessed like this::

  trace.df_event("sched_switch")

Whereas analysis dataframes can be obtained like that::

  # trace.ana.<analysis name>.<analysis method>
  trace.ana.tasks.df_tasks_states()

.. seealso:: See the :class:`~lisa.trace.Trace` documentation for more details.

Available analysis
==================

Dataframes
++++++++++

The majority of these dataframes are time-indexed (and if they aren't, it will
be called out in the docstring). This makes it easy to create dataframe slices
to study specific trace windows.

.. exec::
    from lisa.doc.helpers import get_analysis_list
    print(get_analysis_list("df"))

Plots
+++++

When run in a notebook, these plots will be displayed automatically. By default,
they are also saved in the same directory as your ``trace.dat``

.. exec::
    from lisa.doc.helpers import get_analysis_list
    print(get_analysis_list("plot"))

API
===

Trace
+++++

.. autoclass:: lisa.trace.Trace
   :members:

.. automodule:: lisa.trace
   :members:
   :exclude-members: Trace, TraceParserBase, EventParserBase, TxtTraceParserBase, MetaTxtTraceParser, TxtTraceParser, SimpleTxtTraceParser, HRTxtTraceParser, SysTraceParser, TxtEventParser, CustomFieldsTxtEventParser, PrintTxtEventParser, TrappyTraceParser

Analysis proxy
++++++++++++++

.. automodule:: lisa.analysis.proxy
   :members:

Analysis base class
+++++++++++++++++++

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

rt-app
++++++

.. automodule:: lisa.analysis.rta
   :members:

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

Interactive notebook helper
+++++++++++++++++++++++++++

.. automodule:: lisa.analysis.notebook
   :members:


Trace parsers
+++++++++++++

.. note:: :class:`lisa.trace.Trace` is the class to use to manipulate a trace
    file, trace parsers are backend objects that are usually not
    manipulated by the user.

.. autoclass:: lisa.trace.TraceParserBase
   :members:

.. autoclass:: lisa.trace.EventParserBase
   :members:

.. autoclass:: lisa.trace.TxtTraceParserBase
   :members:

.. autoclass:: lisa.trace.TxtTraceParser
   :members:

.. autoclass:: lisa.trace.MetaTxtTraceParser
   :members:

.. autoclass:: lisa.trace.SimpleTxtTraceParser
   :members:

.. autoclass:: lisa.trace.HRTxtTraceParser
   :members:

.. autoclass:: lisa.trace.SysTraceParser
   :members:

.. autoclass:: lisa.trace.TxtEventParser
   :members:

.. autoclass:: lisa.trace.CustomFieldsTxtEventParser
   :members:

.. autoclass:: lisa.trace.PrintTxtEventParser
   :members:

.. autoclass:: lisa.trace.TrappyTraceParser
   :members:
