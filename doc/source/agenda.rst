.. _agenda:

======
Agenda
======

An agenda specifies what is to be done during a Workload Automation run,
including which workloads will be run, with what configuration, which
instruments and result processors will be enabled, etc. Agenda syntax is
designed to be both succinct and expressive.

Agendas are specified using YAML_ notation. It is recommended that you
familiarize yourself with the linked page.

.. _YAML: http://en.wikipedia.org/wiki/YAML

.. note:: Earlier versions of WA have supported CSV-style agendas. These were
          there to facilitate transition from WA1 scripts. The format was more
          awkward and supported only a limited subset of the features. Support
          for it has now been removed.


Specifying which workloads to run
=================================

The central purpose of an agenda is to specify what workloads to run. A
minimalist agenda contains a single entry at the top level called "workloads"
that maps onto a list of workload names to run:

.. code-block:: yaml

        workloads:
                - dhrystone
                - memcpy
                - cyclictest

This specifies a WA run consisting of ``dhrystone`` followed by ``memcpy``, followed by
``cyclictest`` workloads, and using instruments and result processors specified in
config.py (see :ref:`configuration-specification` section).

.. note:: If you're familiar with YAML, you will recognize the above as a single-key
          associative array mapping onto a list. YAML has two notations for both
          associative arrays and lists: block notation (seen above) and also
          in-line notation. This means that the above agenda can also be
          written in a single line as ::

                workloads: [dhrystone, memcpy, cyclictest]

          (with the list in-lined), or ::

                {workloads: [dhrystone, memcpy, cyclictest]}

          (with both the list and the associative array in-line). WA doesn't
          care which of the notations is used as they all get parsed into the
          same structure by the YAML parser. You can use whatever format you
          find easier/clearer.

Multiple iterations
-------------------

There will normally be some variability in workload execution when running on a
real device. In order to quantify it, multiple iterations of the same workload
are usually performed. You can specify the number of iterations for each
workload by adding ``iterations`` field to the workload specifications (or
"specs"):

.. code-block:: yaml

        workloads:
                - name: dhrystone
                  iterations: 5
                - name: memcpy
                  iterations: 5
                - name: cyclictest
                  iterations: 5

Now that we're specifying both the workload name and the number of iterations in
each spec, we have to explicitly name each field of the spec.

It is often the case that, as in in the example above, you will want to run all
workloads for the same number of iterations. Rather than having to specify it
for each and every spec, you can do with a single entry by adding a ``global``
section to your agenda:

.. code-block:: yaml

        global:
                iterations: 5
        workloads:
                - dhrystone
                - memcpy
                - cyclictest

The global section can contain the same fields as a workload spec. The
fields in the global section will get added to each spec. If the same field is
defined both in global section and in a spec, then the value in the spec will
overwrite the global value. For example, suppose we wanted to run all our workloads
for five iterations, except cyclictest which we want to run for ten (e.g.
because we know it to be particularly unstable). This can be specified like
this:

.. code-block:: yaml

        global:
                iterations: 5
        workloads:
                - dhrystone
                - memcpy
                - name: cyclictest
                  iterations: 10

Again, because we are now specifying two fields for cyclictest spec, we have to
explicitly name them.

Configuring workloads
---------------------

Some workloads accept configuration parameters that modify their behavior. These
parameters are specific to a particular workload and can alter the workload in
any number of ways, e.g. set the duration for which to run, or specify a media
file to be used, etc. The vast majority of workload parameters will have some
default value, so it is only necessary to specify the name of the workload in
order for WA to run it. However, sometimes you want more control over how a
workload runs.

For example, by default, dhrystone will execute 10 million loops across four
threads. Suppose you device has six cores available and you want the workload to
load them all. You also want to increase the total number of loops accordingly
to 15 million. You can specify this using dhrystone's parameters:

.. code-block:: yaml

        global:
                iterations: 5
        workloads:
                - name: dhrystone
                  params:
                        threads: 6
                        mloops: 15
                - memcpy
                - name: cyclictest
                  iterations: 10

.. note:: You can find out what parameters a workload accepts by looking it up
          in the :ref:`Workloads` section. You can also look it up using WA itself
          with "show" command::

                wa show dhrystone

          see the :ref:`Invocation` section for details.

In addition to configuring the workload itself, we can also specify
configuration for the underlying device. This can be done by setting runtime
parameters in the workload spec. For example, suppose we want to ensure the
maximum score for our benchmarks, at the expense of power consumption, by
setting the cpufreq governor to "performance" on cpu0 (assuming all our cores
are in the same DVFS domain and so setting the governor for cpu0 will affect all
cores). This can be done like this:

.. code-block:: yaml

        global:
                iterations: 5
        workloads:
                - name: dhrystone
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - memcpy
                - name: cyclictest
                  iterations: 10


Here, we're specifying ``sysfile_values`` runtime parameter for the device. The
value for this parameter is a mapping (an associative array, in YAML) of file
paths onto values that should be written into those files. ``sysfile_values`` is
the only runtime parameter that is available for any (Linux) device. Other
runtime parameters will depend on the specifics of the device used (e.g. its
CPU cores configuration). I've renamed ``params`` to   ``workload_params`` for
clarity, but that wasn't strictly necessary as ``params`` is interpreted as
``workload_params`` inside a workload spec.

.. note:: ``params`` field is interpreted differently depending on whether it's in a
          workload spec or the global section. In a workload spec, it translates to
          ``workload_params``, in the global section it translates to ``runtime_params``.

Runtime parameters do not automatically reset at the end of workload spec
execution, so all subsequent iterations will also be affected unless they
explicitly change the parameter (in the example above, performance governor will
also be used for ``memcpy`` and ``cyclictest``. There are two ways around this:
either set ``reboot_policy`` WA setting (see :ref:`configuration-specification` section) such that
the device gets rebooted between spec executions, thus being returned to its
initial state, or set the default runtime parameter values in the ``global``
section of the agenda so that they get set for every spec that doesn't
explicitly override them.

.. note:: "In addition to ``runtime_params`` there are also ``boot_params`` that
           work in a similar way, but they get passed to the device when it
           reboots. At the moment ``TC2`` is the only device that defines a boot
           parameter, which is explained in ``TC2`` documentation, so boot
           parameters will not be mentioned further.

IDs and Labels
--------------

It is possible to list multiple specs with the same workload in an agenda. You
may wish to this if you want to run a workload with different parameter values
or under different runtime configurations of the device. The workload name
therefore does not uniquely identify a spec. To be able to distinguish between
different specs (e.g. in reported results), each spec has an ID which is unique
to all specs within an agenda (and therefore with a single WA run). If an ID
isn't explicitly specified using ``id`` field (note that the field name is in
lower case), one will be automatically assigned to the spec at the beginning of
the WA run based on the position of the spec within the list. The first spec
*without an explicit ID* will be assigned ID ``1``, the second spec *without an
explicit ID*  will be assigned ID ``2``, and so forth.

Numerical IDs aren't particularly easy to deal with, which is why it is
recommended that, for non-trivial agendas, you manually set the ids to something
more meaningful (or use labels -- see below). An ID can be pretty much anything
that will pass through the YAML parser. The only requirement is that it is
unique to the agenda. However, is usually better to keep them reasonably short
(they don't need to be *globally* unique), and to stick with alpha-numeric
characters and underscores/dashes. While WA can handle other characters as well,
getting too adventurous with your IDs may cause issues further down the line
when processing WA results (e.g. when uploading them to a database that may have
its own restrictions).

In addition to IDs, you can also specify labels for your workload specs. These
are similar to IDs but do not have the uniqueness restriction. If specified,
labels will be used by some result processes instead of (or in addition to) the
workload name. For example, the ``csv`` result processor will put the label in the
"workload" column of the CSV file.

It is up to you how you chose to use IDs and labels. WA itself doesn't expect
any particular format (apart from uniqueness for IDs). Below is the earlier
example updated to specify explicit IDs and label dhrystone spec to reflect
parameters used.

.. code-block:: yaml

        global:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10


Result Processors and Instrumentation
=====================================

Result Processors
-----------------

Result processors, as the name suggests, handle the processing of results
generated form running workload specs. By default, WA enables a couple of basic
result processors (e.g. one generates a csv file with all scores reported by
workloads), which you can see in ``~/.workload_automation/config.py``. However,
WA has a number of other, more specialized, result processors (e.g. for
uploading to databases). You can list available result processors with
``wa list result_processors`` command. If you want to permanently enable a
result processor, you can add it to your ``config.py``. You can also enable a
result processor for a particular run by specifying it in the ``config`` section
in the agenda. As the name suggests, ``config`` section mirrors the structure of
``config.py``\ (although using YAML rather than Python), and anything that can
be specified in the latter, can also be specified in the former.

As with workloads, result processors may have parameters that define their
behavior. Parameters of result processors are specified a little differently,
however. Result processor parameter values are listed in the config section,
namespaced under the name of the result processor.

For example, suppose we want to be able to easily query the results generated by
the workload specs we've defined so far. We can use ``sqlite`` result processor
to have WA create an sqlite_ database file with the results. By default, this
file will be generated in WA's output directory (at the same level as
results.csv); but suppose we want to store the results in the same file for
every run of the agenda we do. This can be done by specifying an alternative
database file with ``database`` parameter of the result processor:

.. code-block:: yaml

        config:
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

A couple of things to observe here:

- There is no need to repeat the result processors listed in ``config.py``. The
  processors listed in ``result_processors`` entry in the agenda will be used
  *in addition to* those defined in the ``config.py``.
- The database file is specified under "sqlite" entry in the config section.
  Note, however, that this entry alone is not enough to enable the result
  processor, it must be listed in ``result_processors``, otherwise the "sqilte"
  config entry will be ignored.
- The database file must be specified as an absolute path, however it may use
  the user home specifier '~' and/or environment variables.

.. _sqlite: http://www.sqlite.org/


Instrumentation
---------------

WA can enable various "instruments" to be used during workload execution.
Instruments can be quite diverse in their functionality, but the majority of
instruments available in WA today are there to collect additional data (such as
trace) from the device during workload execution. You can view the list of
available instruments by using ``wa list instruments`` command. As with result
processors, a few are enabled by default in the ``config.py`` and additional
ones may be added in the same place, or specified in the agenda using
``instrumentation`` entry.

For example, we can collect core utilisation statistics (for what proportion of
workload execution N cores were utilized above a specified threshold) using
``coreutil`` instrument.

.. code-block:: yaml

        config:
                instrumentation: [coreutil]
                coreutil:
                        threshold: 80
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

Instrumentation isn't "free" and it is advisable not to have too many
instruments enabled at once as that might skew results. For example, you don't
want to have power measurement enabled at the same time as event tracing, as the
latter may prevent cores from going into idle states and thus affecting the
reading collected by the former.

Unlike result processors, instrumentation may be enabled (and disabled -- see below)
on per-spec basis. For example, suppose we want to collect /proc/meminfo from the
device when we run ``memcpy`` workload, but not for the other two. We can do that using
``sysfs_extractor`` instrument, and we will only enable it for ``memcpy``:

.. code-block:: yaml

        config:
                instrumentation: [coreutil]
                coreutil:
                        threshold: 80
                sysfs_extractor:
                        paths: [/proc/meminfo]
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  instrumentation: [sysfs_extractor]
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

As with ``config`` sections, ``instrumentation`` entry in the spec needs only to
list additional instruments and does not need to repeat instruments specified
elsewhere.

.. note:: At present, it is only possible to enable/disable instrumentation  on
          per-spec base. It is *not* possible to provide configuration on
          per-spec basis in the current version of WA (e.g. in our example, it
          is not possible to specify different ``sysfs_extractor`` paths for
          different workloads). This restriction may be lifted in future
          versions of WA.

Disabling result processors and instrumentation
-----------------------------------------------

As seen above, plugins specified with ``instrumentation`` and
``result_processor`` clauses get added to those already specified previously.
Just because an instrument specified in ``config.py`` is not listed in the
``config`` section of the agenda, does not mean it will be disabled. If you do
want to disable an instrument, you can always remove/comment it out from
``config.py``. However that will be introducing a permanent configuration change
to your environment (one that can be easily reverted, but may be just as
easily forgotten). If you want to temporarily disable a result processor or an
instrument for a particular run, you can do that in your agenda by prepending a
tilde (``~``) to its name.

For example, let's say we want to disable ``cpufreq`` instrument enabled in our
``config.py`` (suppose we're going to send results via email and so want to
reduce to total size of the output directory):

.. code-block:: yaml

        config:
                instrumentation: [coreutil, ~cpufreq]
                coreutil:
                        threshold: 80
                sysfs_extractor:
                        paths: [/proc/meminfo]
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  instrumentation: [sysfs_extractor]
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10


Sections
========

It is a common requirement to be able to run the same set of workloads under
different device configurations. E.g. you may want to investigate impact of
changing a particular setting to different values on the benchmark scores, or to
quantify the impact of enabling a particular feature in the kernel. WA allows
this by defining "sections" of configuration with an agenda.

For example, suppose what we really want, is to measure the impact of using
interactive cpufreq governor vs the performance governor on the three
benchmarks. We could create another three workload spec entries similar to the
ones we already have and change the sysfile value being set to "interactive".
However, this introduces a lot of duplication; and what if we  want to change
spec configuration? We would have to change it in multiple places, running the
risk of forgetting one.

A better way is to keep the three workload specs and define a section for each
governor:

.. code-block:: yaml

        config:
                instrumentation: [coreutil, ~cpufreq]
                coreutil:
                        threshold: 80
                sysfs_extractor:
                        paths: [/proc/meminfo]
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        sections:
                - id: perf
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                - id: inter
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: interactive
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  instrumentation: [sysfs_extractor]
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

A section, just like an workload spec, needs to have a unique ID. Apart from
that, a "section" is similar to the ``global`` section we've already seen --
everything that goes into a section will be applied to each workload spec.
Workload specs defined under top-level ``workloads`` entry will be executed for
each of the sections listed under ``sections``.

.. note:: It is also possible to have a ``workloads`` entry within a section,
          in which case, those workloads will only be executed for that specific
          section.

In order to maintain the uniqueness requirement of workload spec IDs, they will
be namespaced under each section by prepending the section ID to the spec ID
with an under score. So in the agenda above, we no longer have a workload spec
with ID ``01_dhry``, instead there are two specs with IDs ``perf_01_dhry`` and
``inter_01_dhry``.

Note that the ``global`` section still applies to every spec in the agenda. So
the precedence order is -- spec settings override section settings, which in
turn override global settings.


Other Configuration
===================

.. _configuration_in_agenda:

As mentioned previously, ``config`` section in an agenda can contain anything
that can be defined in ``config.py`` (with Python syntax translated to the
equivalent YAML). Certain configuration (e.g. ``run_name``) makes more sense
to define in an agenda than a config file. Refer to the
:ref:`configuration-specification` section for details.

.. code-block:: yaml

        config:
                project: governor_comparison
                run_name: performance_vs_interactive

                device: generic_android
                reboot_policy: never

                instrumentation: [coreutil, ~cpufreq]
                coreutil:
                        threshold: 80
                sysfs_extractor:
                        paths: [/proc/meminfo]
                result_processors: [sqlite]
                sqlite:
                        database: ~/my_wa_results.sqlite
        global:
                iterations: 5
        sections:
                - id: perf
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                - id: inter
                  runtime_params:
                        sysfile_values:
                                /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: interactive
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  instrumentation: [sysfs_extractor]
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

