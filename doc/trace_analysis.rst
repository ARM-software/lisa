.. _analysis-page:

*********************
Kernel trace analysis
*********************

Introduction
============

LISA comes with a plethora of analysis functions based on `Ftrace
<https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces. We
convert the trace events into dataframes (:class:`polars.LazyFrame` and
:class:`pandas.DataFrame` are currently supported).

Trace
=====

Our :class:`~lisa.trace.Trace` takes an Ftrace ``trace.dat`` file as input
(other text-based formats are also accepted, but mileage may vary since they
are ambiguous), and provides access to both the raw trace events, as well as
some dataframes (:class:`polars.LazyFrame` and :class:`pandas.DataFrame`) built
from analysing and aggregating trace events.

You can create one like so::

  trace = Trace("path/to/trace.dat")

Raw trace events can be accessed like this::

  trace.df_event("sched_switch")

Whereas analysis dataframes can be obtained like that::

  # trace.ana.<analysis name>.<analysis method>
  trace.ana.tasks.df_tasks_states()

Switching to :mod:`polars` can be done with::

  trace = Trace("path/to/trace.dat", df_fmt='polars-lazyframe')

Or from an existing :class:`~lisa.trace.Trace` object::

  trace = Trace("path/to/trace.dat")
  trace = trace.get_view(df_fmt='polars-lazyframe')

Then all the dataframe APIs will return :class:`polars.LazyFrame` instances
instead of :class:`pandas.DataFrame`.


Here are the main entry points in the trace analysis APIs:

* Trace manipulation class: :class:`lisa.trace.Trace`
* Trace analysis package: :mod:`lisa.analysis`
* Trace analysis base classes: :mod:`lisa.analysis.base`


Available analysis
==================

Dataframes
++++++++++

The majority of these dataframes are time-indexed (and if they aren't, it will
be called out in the docstring). This makes it easy to create dataframe slices
to study specific trace windows.

.. exec::
    from lisa._doc.helpers import get_analysis_list
    print(get_analysis_list("df"))


Gallery
+++++++

.. exec::
    # Get the state exposed by lisa-exec-state sphinx hook
    plots = state.plots

    if plots:
        from itertools import starmap

        from lisa.analysis.base import TraceAnalysisBase
        from lisa.utils import get_obj_name, groupby, get_parent_namespace

        from lisa._doc.helpers import ana_invocation


        def make_entry(f, rst_fig):
            name = get_obj_name(f, style='rst', abbrev=True)
            rst_fig = rst_fig or 'No plot available'
            invocation = ana_invocation(f)
            return f'\n\n{name}\n{"." * len(name)}\n\n{invocation}\n\n{rst_fig}'

        def make_sections(section, entries):
            entries = sorted(starmap(make_entry, entries))
            entries = '\n\n'.join(entries)
            return f'{section}\n{"-" * len(section)}\n\n{entries}'

        def key(item):
            f, fig  = item
            ns = get_parent_namespace(f)
            assert isinstance(ns, type)
            assert issubclass(ns, TraceAnalysisBase)
            return ns.name

        sections = groupby(plots.items(), key=key)
        sections = sorted(starmap(make_sections, sections))
        sections = '\n\n'.join(sections)

        print(sections)

    else:
        print('No plots available')

