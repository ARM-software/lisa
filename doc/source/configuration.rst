.. _configuration-specification:

=============
Configuration
=============

In addition to specifying run execution parameters through an agenda, the
behavior of WA can be modified through configuration file(s). The default
configuration file is ``~/.workload_automation/config.py``  (the location can be
changed by setting ``WA_USER_DIRECTORY`` environment variable, see :ref:`envvars`
section below). This file will be
created when you first run WA if it does not already exist. This file must
always exist and will always be loaded. You can add to or override the contents
of that file on invocation of Workload Automation by specifying an additional
configuration file using ``--config`` option.

The config file is just a Python source file, so it can contain any valid Python
code (though execution of arbitrary code through the config file is
discouraged). Variables with specific names  will be picked up by the framework
and used to modify the behavior of Workload automation.

.. note:: As of version 2.1.3 is also possible to specify the following
          configuration in the agenda. See :ref:`configuration in an agenda <configuration_in_agenda>`\ .


.. _available_settings:

Available Settings
==================

.. note:: Plugins such as workloads, instrumentation or result processors
          may also pick up certain settings from this file, so the list below is
          not exhaustive. Please refer to the documentation for the specific
          plugins to see what settings they accept.

.. confval:: device

   This setting defines what specific Device subclass will be used to interact
   the connected device. Obviously, this must match your setup.

.. confval:: device_config

   This must be a Python dict containing setting-value mapping for the
   configured :rst:dir:`device`. What settings and values are valid is specific
   to each device. Please refer to the documentation for your device.

.. confval:: reboot_policy

   This defines when during execution of a run the Device will be rebooted. The
   possible values are:

   ``"never"``
      The device will never be rebooted.
   ``"initial"``
      The device will be rebooted when the execution first starts, just before
      executing the first workload spec.
   ``"each_spec"``
      The device will be rebooted before running a new workload spec.
      Note: this acts the same as each_iteration when execution order is set to by_iteration
   ``"each_iteration"``
      The device will be rebooted before each new iteration.

   .. seealso::

      :doc:`execution_model`

.. confval:: execution_order

   Defines the order in which the agenda spec will be executed. At the moment,
   the following execution orders are supported:

   ``"by_iteration"``
     The first iteration of each workload spec is executed one after the other,
     so all workloads are executed before proceeding on to the second iteration.
     E.g. A1 B1 C1 A2 C2 A3. This is the default if no order is explicitly specified.

     In case of multiple sections, this will spread them out, such that specs
     from the same section are further part. E.g. given sections X and Y, global
     specs A and B, and two iterations, this will run ::

                     X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2

   ``"by_section"``
     Same  as ``"by_iteration"``, however this will group specs from the same
     section together, so given sections X and Y, global specs A and B, and two iterations,
     this will run ::

             X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2

   ``"by_spec"``
     All iterations of the first spec are executed before moving on to the next
     spec. E.g. A1 A2 A3 B1 C1 C2 This may also be specified as ``"classic"``,
     as this was the way workloads were executed in earlier versions of WA.

   ``"random"``
     Execution order is entirely random.

   Added in version 2.1.5.


.. confval:: retry_on_status

   This is list of statuses on which a job will be cosidered to have failed and
   will be automatically retried up to ``max_retries`` times. This defaults to
   ``["FAILED", "PARTIAL"]`` if not set. Possible values are:

   ``"OK"``
   This iteration has completed and no errors have been detected

   ``"PARTIAL"``
   One or more instruments have failed (the iteration may still be running).

   ``"FAILED"``
   The workload itself has failed.

   ``"ABORTED"``
   The user interupted the workload

.. confval:: max_retries

   The maximum number of times failed jobs will be retried before giving up. If
   not set, this will default to ``3``.

   .. note:: this number does not include the original attempt

.. confval:: instrumentation

   This should be a list of instruments to be enabled during run execution.
   Values must be names of available instruments. Instruments are used to
   collect additional data, such as energy measurements or execution time,
   during runs.

   .. seealso::

      :doc:`api/wlauto.instrumentation`

.. confval:: result_processors

   This should be a list of result processors to be enabled during run execution.
   Values must be names of available result processors. Result processor define
   how data is output from WA.

   .. seealso::

      :doc:`api/wlauto.result_processors`

.. confval:: logging

   A dict that contains logging setting. At the moment only three settings are
   supported:

   ``"file format"``
      Controls how logging output appears in the run.log file in the output
      directory.
   ``"verbose format"``
      Controls how logging output appear on the console when ``--verbose`` flag
      was used.
   ``"regular format"``
      Controls how logging output appear on the console when ``--verbose`` flag
      was not used.

   All three values should be Python `old-style format strings`_ specifying which
   `log record attributes`_ should be displayed.

.. confval:: remote_assets_path

   Path to the local mount of a network assets repository. See
   :ref:`assets_repository`.


There are also a couple of settings are used to provide additional metadata
for a run. These may get picked up by instruments or result processors to
attach  context to results.

.. confval:: project

   A string naming the project for which data is being collected. This may be
   useful, e.g. when uploading data to a shared database that is populated from
   multiple projects.

.. confval:: project_stage

   A dict or a string that allows adding additional identifier. This is may be
   useful for long-running projects.

.. confval:: run_name

   A string that labels the WA run that is bing performed. This would typically
   be set in the ``config`` section of an agenda (see
   :ref:`configuration in an agenda <configuration_in_agenda>`) rather than in the config file.

.. _old-style format strings: http://docs.python.org/2/library/stdtypes.html#string-formatting-operations
.. _log record attributes: http://docs.python.org/2/library/logging.html#logrecord-attributes


.. _envvars:

Environment Variables
=====================

In addition to standard configuration described above, WA behaviour can be
altered through environment variables. These can determine where WA looks for
various assets when it starts.

.. confval:: WA_USER_DIRECTORY

   This is the location WA will look for config.py, inustrumentation , and it
   will also be used for local caches, etc. If this variable is not set, the
   default location is ``~/.workload_automation`` (this is created when WA
   is installed).

   .. note:: This location **must** be writable by the user who runs WA.


.. confval:: WA_PLUGIN_PATHS

   By default, WA will look for plugins in its own package and in
   subdirectories under ``WA_USER_DIRECTORY``. This environment variable can
   be used specify a colon-separated list of additional locations WA should
   use to look for plugins.
