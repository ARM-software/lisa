.. _agenda:

Defining Experiments With an Agenda
===================================

An agenda specifies what is to be done during a Workload Automation run,
including which workloads will be run, with what configuration, which
augmentations will be enabled, etc. Agenda syntax is designed to be both
succinct and expressive.

Agendas are specified using YAML_ notation. It is recommended that you
familiarize yourself with the linked page.

.. _YAML: http://en.wikipedia.org/wiki/YAML

Specifying which workloads to run
---------------------------------

The central purpose of an agenda is to specify what workloads to run. A
minimalist agenda contains a single entry at the top level called "workloads"
that maps onto a list of workload names to run:

.. code-block:: yaml

        workloads:
                - dhrystone
                - memcpy
                - rt_app

This specifies a WA run consisting of ``dhrystone`` followed by ``memcpy``, followed by
``rt_app`` workloads, and using the augmentations specified in
config.yaml (see :ref:`configuration-specification` section).

.. note:: If you're familiar with YAML, you will recognize the above as a single-key
          associative array mapping onto a list. YAML has two notations for both
          associative arrays and lists: block notation (seen above) and also
          in-line notation. This means that the above agenda can also be
          written in a single line as ::

                workloads: [dhrystone, memcpy, rt-app]

          (with the list in-lined), or ::

                {workloads: [dhrystone, memcpy, rt-app]}

          (with both the list and the associative array in-line). WA doesn't
          care which of the notations is used as they all get parsed into the
          same structure by the YAML parser. You can use whatever format you
          find easier/clearer.

.. note:: WA plugin names are case-insensitive, and dashes (``-``) and
          underscores (``_``) are treated identically. So all of the following
          entries specify the same workload: ``rt_app``, ``rt-app``, ``RT-app``.

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
for each and every spec, you can do with a single entry by adding `iterations`
to your ``config`` section in your agenda:

.. code-block:: yaml

        config:
                iterations: 5
        workloads:
                - dhrystone
                - memcpy
                - cyclictest

If the same field is defined both in config section and in a spec, then the
value in the spec will overwrite the  value. For example, suppose we
wanted to run all our workloads for five iterations, except cyclictest which we
want to run for ten (e.g. because we know it to be particularly unstable). This
can be specified like this:

.. code-block:: yaml

        config:
                iterations: 5
        workloads:
                - dhrystone
                - memcpy
                - name: cyclictest
                  iterations: 10

Again, because we are now specifying two fields for cyclictest spec, we have to
explicitly name them.

Configuring Workloads
---------------------

Some workloads accept configuration parameters that modify their behaviour. These
parameters are specific to a particular workload and can alter the workload in
any number of ways, e.g. set the duration for which to run, or specify a media
file to be used, etc. The vast majority of workload parameters will have some
default value, so it is only necessary to specify the name of the workload in
order for WA to run it. However, sometimes you want more control over how a
workload runs.

For example, by default, dhrystone will execute 10 million loops across four
threads. Suppose your device has six cores available and you want the workload to
load them all. You also want to increase the total number of loops accordingly
to 15 million. You can specify this using dhrystone's parameters:

.. code-block:: yaml

        config:
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
          in the :ref:`Workloads` section or using WA itself with "show"
          command::

                wa show dhrystone

          see the :ref:`Invocation` section for details.

In addition to configuring the workload itself, we can also specify
configuration for the underlying device which can be done by setting runtime
parameters in the workload spec. Explicit runtime parameters have been exposed for
configuring cpufreq, hotplug and cpuidle. For more detailed information on Runtime
Parameters see the :ref:`runtime parameters <runtime-parameters>` section. For
example, suppose we want to ensure the maximum score for our benchmarks, at the
expense of power consumption so we want to set the cpufreq governor to
"performance" and enable all of the cpus on the device, (assuming there are 8
cpus available), which can be done like this:

.. code-block:: yaml

        config:
                iterations: 5
        workloads:
                - name: dhrystone
                  runtime_params:
                        governor: performance
                        num_cores: 8
                  workload_params:
                        threads: 6
                        mloops: 15
                - memcpy
                - name: cyclictest
                  iterations: 10


I've renamed ``params`` to   ``workload_params`` for clarity,
but that wasn't strictly necessary as ``params`` is interpreted as
``workload_params`` inside a workload spec.

Runtime parameters do not automatically reset at the end of workload spec
execution, so all subsequent iterations will also be affected unless they
explicitly change the parameter (in the example above, performance governor will
also be used for ``memcpy`` and ``cyclictest``. There are two ways around this:
either set ``reboot_policy`` WA setting (see :ref:`configuration-specification`
section) such that the device gets rebooted between job executions, thus being
returned to its initial state, or set the default runtime parameter values in
the ``config`` section of the agenda so that they get set for every spec that
doesn't explicitly override them.

If additional configuration of the device is required which are not exposed via
the built in runtime parameters, you can write a value to any file exposed on
the device using ``sysfile_values``, for example we could have also performed
the same configuration manually (assuming we have a big.LITTLE system and our
cores 0-3 and 4-7 are in 2 separate DVFS domains and so setting the governor for
cpu0 and cpu4 will affect all our cores) e.g.

.. code-block:: yaml


        config:
                iterations: 5
        workloads:
                - name: dhrystone
                runtime_params:
                        sysfile_values:
                            /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor: performance
                            /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor: performance
                            /sys/devices/system/cpu/cpu0/online: 1
                            /sys/devices/system/cpu/cpu1/online: 1
                            /sys/devices/system/cpu/cpu2/online: 1
                            /sys/devices/system/cpu/cpu3/online: 1
                            /sys/devices/system/cpu/cpu4/online: 1
                            /sys/devices/system/cpu/cpu5/online: 1
                            /sys/devices/system/cpu/cpu6/online: 1
                            /sys/devices/system/cpu/cpu7/online: 1
                workload_params:
                        threads: 6
                        mloops: 15
            - memcpy
            - name: cyclictest
                iterations: 10

Here, we're specifying a ``sysfile_values`` runtime parameter for the device.
For more information please see :ref:`setting sysfiles <setting-sysfiles>`.

APK Workloads
^^^^^^^^^^^^^

WA has various resource getters that can be configured to locate APK files but
for most people APK files should be kept in the
``$WA_USER_DIRECTORY/dependencies/SOME_WORKLOAD/`` directory. (by default
``~/.workload_automation/dependencies/SOME_WORKLOAD/``). The
``WA_USER_DIRECTORY`` environment variable can be used to change the location of
this directory. The APK files need to be put into the corresponding directories for
the workload they belong to. The name of the file can be anything but as
explained below may need to contain certain pieces of information.

All ApkWorkloads have parameters that affect the way in which APK files are
resolved, ``exact_abi``, ``force_install`` and ``prefer_host_package``. Their
exact behaviours are outlined below.

:exact_abi: If this setting is enabled WA's resource resolvers will look for the
   devices ABI with any native code present in the apk. By default this setting
   is disabled since most apks will work across all devices. You may wish to
   enable this feature when working with devices that support multiple ABI's
   (like 64-bit devices that can run 32-bit APK files) and are specifically
   trying to test one or the other.

:force_install: If this setting is enabled WA will *always* use the APK file on
   the host, and re-install it on every iteration. If there is no APK on the
   host that is a suitable version and/or ABI for the workload WA will error
   when ``force_install`` is enabled.

:prefer_host_package: This parameter is used to specify a preference over host
   or target versions of the app. When set to ``True`` WA will prefer the host
   side version of the APK. It will check if the host has the APK and whether it
   meets the version requirements of the workload. If so, and the target also
   already has same version nothing will be done, otherwise WA will overwrite
   the targets installed application with the host version. If the host is
   missing the APK or it does not meet version requirements WA will fall back to
   the app on the target if present and is a suitable version. When this
   parameter is set to ``False`` WA will prefer to use the version already on
   the target if it meets the workloads version requirements. If it does not it
   will fall back to searching the host for the correct version. In both modes
   if neither the host nor target have a suitable version, WA will produce and
   error and will not run the workload.

:version: This parameter is used to specify which version of uiautomation for
   the workload is used. In some workloads e.g. ``geekbench`` multiple versions
   with drastically different UI's are supported. A APKs version will be
   automatically extracted therefore it is possible to have multiple apks for
   different versions of a workload present on the host and select between which
   is used for a particular job by specifying the relevant version in your
   :ref:`agenda <agenda>`.

:variant_name: Some workloads use variants of APK files, this is usually the
   case with web browser APK files, these work in exactly the same way as the
   version.


IDs and Labels
--------------

It is possible to list multiple specs with the same workload in an agenda. You
may wish to do this if you want to run a workload with different parameter values
or under different runtime configurations of the device. The workload name
therefore does not uniquely identify a spec. To be able to distinguish between
different specs (e.g. in reported results), each spec has an ID which is unique
to all specs within an agenda (and therefore with a single WA run). If an ID
isn't explicitly specified using ``id`` field (note that the field name is in
lower case), one will be automatically assigned to the spec at the beginning of
the WA run based on the position of the spec within the list. The first spec
*without an explicit ID* will be assigned ID ``wk1``, the second spec *without an
explicit ID*  will be assigned ID ``wk2``, and so forth.

Numerical IDs aren't particularly easy to deal with, which is why it is
recommended that, for non-trivial agendas, you manually set the ids to something
more meaningful (or use labels -- see below). An ID can be pretty much anything
that will pass through the YAML parser. The only requirement is that it is
unique to the agenda. However, is usually better to keep them reasonably short
(they don't need to be *globally* unique), and to stick with alpha-numeric
characters and underscores/dashes. While WA can handle other characters as well,
getting too adventurous with your IDs may cause issues further down the line
when processing WA output (e.g. when uploading them to a database that may have
its own restrictions).

In addition to IDs, you can also specify labels for your workload specs. These
are similar to IDs but do not have the uniqueness restriction. If specified,
labels will be used by some output processes instead of (or in addition to) the
workload name. For example, the ``csv`` output processor will put the label in the
"workload" column of the CSV file.

It is up to you how you chose to use IDs and labels. WA itself doesn't expect
any particular format (apart from uniqueness for IDs). Below is the earlier
example updated to specify explicit IDs and label dhrystone spec to reflect
parameters used.

.. code-block:: yaml

        config:
                iterations: 5
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  runtime_params:
                        cpu0_governor: performance
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10

.. _using-classifiers:

Classifiers
------------

Classifiers can be used in 2 distinct ways, the first use is being supplied in
an agenda as a set of key-value pairs which can be used to help identify sub-tests
of a run, for example if you have multiple sections in your agenda running
your workloads at different frequencies you might want to set a classifier
specifying which frequencies are being used. These can then be utilized later,
for example with the ``csv`` :ref:`output processor <output-processors>` with
``use_all_classifiers`` set to ``True`` and this will add additional columns to
the output file for each of the classifier keys that have been specified
allowing for quick comparison.

An example agenda is shown here:

.. code-block:: yaml

        config:
            augmentations:
                - csv
            iterations: 1
            device: generic_android
            csv:
                use_all_classifiers: True
        sections:
            - id: max_speed
              runtime_parameters:
                  frequency: 1700000
              classifiers:
                  freq: 1700000
            - id: min_speed
              runtime_parameters:
                  frequency: 200000
              classifiers:
                  freq: 200000
        workloads:
        -   name: recentfling

The other way that they can used is by being automatically added by some
workloads to identify their results metrics and artifacts. For example some
workloads perform multiple tests with the same execution run and therefore will
use metrics to differentiate between them, e.g. the ``recentfling`` workload
will use classifiers to distinguish between which loop a particular result is
for or whether it is an average across all loops ran.

The output from the agenda above will produce a csv file similar to what is
shown below. Some columns have been omitted for clarity however as can been seen
the custom **frequency** classifier column has been added and populated, along
with the **loop** classifier added by the workload.

::

 id              | workload      | metric                    | freq      | loop    | value ‖
 max_speed-wk1   | recentfling   | 90th Percentile           | 1700000   | 1       | 8     ‖
 max_speed-wk1   | recentfling   | 95th Percentile           | 1700000   | 1       | 9     ‖
 max_speed-wk1   | recentfling   | 99th Percentile           | 1700000   | 1       | 16    ‖
 max_speed-wk1   | recentfling   | Jank                      | 1700000   | 1       | 11    ‖
 max_speed-wk1   | recentfling   | Jank%                     | 1700000   | 1       | 1     ‖
 # ...
 max_speed-wk1   | recentfling   | Jank                      | 1700000   | 3       | 1     ‖
 max_speed-wk1   | recentfling   | Jank%                     | 1700000   | 3       | 0     ‖
 max_speed-wk1   | recentfling   | Average 90th Percentqile  | 1700000   | Average | 7     ‖
 max_speed-wk1   | recentfling   | Average 95th Percentile   | 1700000   | Average | 8     ‖
 max_speed-wk1   | recentfling   | Average 99th Percentile   | 1700000   | Average | 14    ‖
 max_speed-wk1   | recentfling   | Average Jank              | 1700000   | Average | 6     ‖
 max_speed-wk1   | recentfling   | Average Jank%             | 1700000   | Average | 0     ‖
 min_speed-wk1   | recentfling   | 90th Percentile           | 200000    | 1       | 7     ‖
 min_speed-wk1   | recentfling   | 95th Percentile           | 200000    | 1       | 8     ‖
 min_speed-wk1   | recentfling   | 99th Percentile           | 200000    | 1       | 14    ‖
 min_speed-wk1   | recentfling   | Jank                      | 200000    | 1       | 5     ‖
 min_speed-wk1   | recentfling   | Jank%                     | 200000    | 1       | 0     ‖
 # ...
 min_speed-wk1   | recentfling   | Jank                      | 200000    | 3       | 5     ‖
 min_speed-wk1   | recentfling   | Jank%                     | 200000    | 3       | 0     ‖
 min_speed-wk1   | recentfling   | Average 90th Percentile   | 200000    | Average | 7     ‖
 min_speed-wk1   | recentfling   | Average 95th Percentile   | 200000    | Average | 8     ‖
 min_speed-wk1   | recentfling   | Average 99th Percentile   | 200000    | Average | 13    ‖
 min_speed-wk1   | recentfling   | Average Jank              | 200000    | Average | 4     ‖
 min_speed-wk1   | recentfling   | Average Jank%             | 200000    | Average | 0     ‖



.. _sections:

Sections
--------

It is a common requirement to be able to run the same set of workloads under
different device configurations. E.g. you may want to investigate the impact of
changing a particular setting to different values on the benchmark scores, or to
quantify the impact of enabling a particular feature in the kernel. WA allows
this by defining "sections" of configuration with an agenda.

For example, suppose that we want to measure the impact of using 3 different
cpufreq governors on 2 benchmarks. We could create 6 separate workload specs
and set the governor runtime parameter for each entry. However, this
introduces a lot of duplication; and what if we want to change spec
configuration? We would have to change it in multiple places, running the risk
of forgetting one.

A better way is to keep the two workload specs and define a section for each
governor:

.. code-block:: yaml

        config:
                iterations: 5
                augmentations:
                    - ~cpufreq
                    - csv
                sysfs_extractor:
                        paths: [/proc/meminfo]
                csv:
                    use_all_classifiers: True
        sections:
                - id: perf
                  runtime_params:
                        cpu0_governor: performance
                - id: inter
                  runtime_params:
                        cpu0_governor: interactive
                - id: sched
                  runtime_params:
                        cpu0_governor: sched
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  augmentations: [sysfs_extractor]

A section, just like an workload spec, needs to have a unique ID. Apart from
that, a "section" is similar to the ``config`` section we've already seen --
everything that goes into a section will be applied to each workload spec.
Workload specs defined under top-level ``workloads`` entry will be executed for
each of the sections listed under ``sections``.

.. note:: It is also possible to have a ``workloads`` entry within a section,
          in which case, those workloads will only be executed for that specific
          section.

In order to maintain the uniqueness requirement of workload spec IDs, they will
be namespaced under each section by prepending the section ID to the spec ID
with a dash. So in the agenda above, we no longer have a workload spec
with ID ``01_dhry``, instead there are two specs with IDs ``perf-01-dhry`` and
``inter-01_dhry``.

Note that the ``config`` section still applies to every spec in the agenda. So
the precedence order is -- spec settings override section settings, which in
turn override global settings.


.. _section-groups:

Section Groups
---------------

Section groups are a way of grouping sections together and are used to produce a
cross product of each of the different groups. This can be useful when you want
to run a set of experiments with all the available combinations without having
to specify each combination manually.

For example if we want to investigate the differences between running the
maximum and minimum frequency with both the maximum and minimum number of cpus
online, we can create an agenda as follows:

.. code-block:: yaml

        sections:
          - id: min_freq
          runtime_parameters:
              freq: min
          group: frequency
         - id: max_freq
          runtime_parameters:
              freq: max
          group: frequency

         - id: min_cpus
           runtime_parameters:
              cpus: 1
          group: cpus
         - id: max_cpus
           runtime_parameters:
              cpus: 8
          group: cpus

        workloads:
        -  dhrystone

This will results in 8 jobs being generated for each of the possible combinations.

::

      min_freq-min_cpus-wk1 (dhrystone)
      min_freq-max_cpus-wk1 (dhrystone)
      max_freq-min_cpus-wk1 (dhrystone)
      max_freq-max_cpus-wk1 (dhrystone)
      min_freq-min_cpus-wk1 (dhrystone)
      min_freq-max_cpus-wk1 (dhrystone)
      max_freq-min_cpus-wk1 (dhrystone)
      max_freq-max_cpus-wk1 (dhrystone)

Each of the generated jobs will have :ref:`classifiers <classifiers>` for
each group and the associated id automatically added.

.. code-block:: python

      # ...
      print('Job ID: {}'.format(job.id))
      print('Classifiers:')
      for k, v in job.classifiers.items():
          print('  {}: {}'.format(k, v))

      Job ID: min_freq-min_cpus-no_idle-wk1
      Classifiers:
          frequency: min_freq
          cpus: min_cpus


.. _augmentations:

Augmentations
--------------

Augmentations are plugins that augment the execution of workload jobs with
additional functionality; usually, that takes the form of generating additional
metrics and/or artifacts, such as traces or logs. There are two types of
augmentations:

Instruments
        These "instrument" a WA run in order to change it's behaviour (e.g.
        introducing delays between successive job executions), or collect
        additional measurements (e.g. energy usage). Some instruments may depend
        on particular features being enabled on the target (e.g. cpufreq), or
        on additional hardware (e.g. energy probes).

Output processors
        These post-process metrics and artifacts generated by workloads or
        instruments, as well as target metadata collected by WA, in order to
        generate additional metrics and/or artifacts (e.g. generating statistics
        or reports). Output processors are also used to export WA output
        externally (e.g. upload to a database).

The main practical difference between instruments and output processors, is that
the former rely on an active connection to the target to function, where as the
latter only operated on previously collected results and metadata. This means
that output processors can run "off-line" using ``wa process`` command.

Both instruments and output processors are configured in the same way in the
agenda, which is why they are grouped together into "augmentations".
Augmentations are enabled by listing them under ``augmentations`` entry in a
config file or ``config`` section of the agenda.

.. code-block:: yaml

        config:
                augmentations: [trace-cmd]

The code above illustrates an agenda entry to enabled ``trace-cmd`` instrument.

If your have multiple ``augmentations`` entries (e.g. both, in your config file
and in the agenda), then they will be combined, so that the final  set of
augmentations for the run  will be their union.

.. note:: WA2 did not have have augmentationts, and instead supported
          "instrumentation" and "result_processors" as distinct configuration
          enetries. For compantibility, these entries are still supported in
          WA3, however they should be considered to be depricated, and their
          use is discouraged.


Configuring augmentations
^^^^^^^^^^^^^^^^^^^^^^^^^

Most augmentations will take parameters that modify their behavior. Parameters
available for a particular augmentation can be viewed using ``wa show
<augmentation name>`` command. This will also show the default values used.
Values for these parameters can be specified by creating an entry with the
augmentation's name, and specifying parameter values under it.

.. code-block:: yaml

        config:
                augmentations: [trace-cmd]
                trace-cmd:
                        events: ['sched*', 'power*', irq]
                        buffer_size: 100000

The code above specifies values for ``events`` and ``buffer_size`` parameters
for the ``trace-cmd`` instrument, as well as enabling it.

You may specify configuration for the same augmentation in multiple locations
(e.g. your config file and the config section of the agenda). These entries will
be combined to form the final configuration for the augmentation used during the
run. If different values for the same parameter are present in multiple entries,
the ones "more specific" to a particular run will be used (e.g. values in the
agenda will override those in the config file).

.. note:: Creating an entry for an augmentation alone does not enable it! You
          **must** list it under ``augmentations`` in order for it to be enabed
          for a run. This makes it easier to quickly enabled and diable
          augmentations with complex configurations, and also allows defining
          "static" configuation in top-level config, without actually enabling
          the augmentation for all runs.


Disabling augmentations
^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, you may wish to disable an augmentation for a particular run, but you
want to keep it enabled in general. You *could* modify your config file to
temporarily disable it. However, you must then remember to re-enable it
afterwards. This could be inconvenient and error prone, especially if you're
running multiple experiments in parallel and only want to disable the
augmentation for one of them.

Instead, you can explicitly disable augmentation by specifying its name prefixed
with a tilde (``~``) inside ``augumentations``.

.. code-block:: yaml

        config:
                augmentations: [trace-cmd, ~cpufreq]

The code above enables ``trace-cmd`` instrument and disables ``cpufreq``
instrument (which is enabled in the default config).

If you want to start configuration for an experiment form a "blank slate" and
want to disable all previously-enabled augmentations, without necessarily
knowing what they are, you can use the special ``~~`` entry.

.. code-block:: yaml

        config:
                augmentations: [~~, trace-cmd, csv]

The code above disables all augmentations enabled up to that point, and enabled
``trace-cmd`` and ``csv`` for this run.

.. note:: The ``~~`` only disables augmentations from previously-processed
          sources. Its ordering in the list does not matter. For example,
          specifying ``augmentations: [trace-cmd, ~~, csv]`` will have exactly
          the same effect as above -- i.e. both trace-cmd *and* csv will be
          enabled.

Workload-specific augmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to enable or disable (but not configure) augmentations at
workload or section level, as well as in the global config, in which case, the
augmentations would only be enabled/disabled for that workload/section. If the
same augmentation is enabled at one level and disabled at another, as with all
WA configuration, the more specific settings will take precedence over the less
specific ones (i.e. workloads override sections that, in turn, override global
config).


Augmentations Example
^^^^^^^^^^^^^^^^^^^^^


.. code-block:: yaml

        config:
                augmentations: [~~, fps]
                trace-cmd:
                        events: ['sched*', 'power*', irq]
                        buffer_size: 100000
                file_poller:
                        files:
                                - /sys/class/thermal/thermal_zone0/temp
        sections:
                - classifers:
                        type: energy
                augmentations: [energy_measurement]
                - classifers:
                        type: trace
                augmentations: [trace-cmd, file_poller]
        workloads:
                - gmail
                - geekbench
                - googleplaybooks
                - name: dhrystone
                  augmentations: [~fps]

The example above shows an experiment that runs a number of workloads in order
to evaluate their thermal impact and energy usage. All previously-configured
augmentations are disabled with ``~~``, so that only configuration specified in
this agenda is enabled. Since most of the workloads are "productivity" use cases
that do not generate their own metrics, ``fps`` instrument is enabled to get
some meaningful performance metrics for them; the only exception is
``dhrystone`` which is a benchmark that reports its own metrics and has not GUI,
so the instrument is disabled for it using ``~fps``.

Each workload will be run in two configurations: once, to collect energy
measurements, and once to collect thermal data and kernel trace. Trace can give
insight into why a workload is using more or less energy than expected, but it
can be relatively intrusive and might impact absolute energy and performance
metrics, which is why it is collected separately. Classifiers_ are used to
separate metrics from the two configurations in the results.

.. _other-agenda-configuration:

Other Configuration
-------------------

.. _configuration_in_agenda:

As mentioned previously, ``config`` section in an agenda can contain anything
that can be defined in ``config.yaml``. Certain configuration (e.g. ``run_name``)
makes more sense to define in an agenda than a config file. Refer to the
:ref:`configuration-specification` section for details.

.. code-block:: yaml

        config:
                project: governor_comparison
                run_name: performance_vs_interactive

                device: generic_android
                reboot_policy: never

                iterations: 5
                augmentations:
                    - ~cpufreq
                    - csv
                sysfs_extractor:
                        paths: [/proc/meminfo]
                csv:
                    use_all_classifiers: True
        sections:
                - id: perf
                  runtime_params:
                        sysfile_values:
                        cpu0_governor: performance
                - id: inter
                  runtime_params:
                        cpu0_governor: interactive
        workloads:
                - id: 01_dhry
                  name: dhrystone
                  label: dhrystone_15over6
                  workload_params:
                        threads: 6
                        mloops: 15
                - id: 02_memc
                  name: memcpy
                  augmentations: [sysfs_extractor]
                - id: 03_cycl
                  name: cyclictest
                  iterations: 10
