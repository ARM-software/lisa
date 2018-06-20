.. _agenda-reference:

Agenda
------


An agenda can be thought of as defining an experiment as it specifies what is to
be done during a Workload Automation run. This includes which workloads will be
run, with what configuration and which augmentations will be enabled, etc.

Agenda syntax is designed to be both succinct and expressive and are written
using YAML notation.

There are three valid top level entries which are: ``"config"``, ``"workloads"``,
``"sections"``.


config
^^^^^^^

This section is used to provide overall configuration for WA and its run. The
``config`` section of an agenda will be merged with any other configuration
files provided (including the default config file) and merged with the most
specific configuration taking precedence (see :ref:`here <config-merging>` for
more information.

Within this section there are multiple distinct types of configuration that can be
provided.

Run Config
~~~~~~~~~~

The first is to configure the behaviour of WA and how a run as a
whole will behave. The most common that you may want to specify are:

- :confval:`device` - The name of the 'device' that you wish to perform the run
  on. This name is a combination of a devlib
  `Platform <http://devlib.readthedocs.io/en/latest/platform.html>`_ and
  `Target <http://devlib.readthedocs.io/en/latest/target.html>`_.
  To see the available options please use ``wa list targets``.
- :confval:`device_config` - The is a dict mapping allowing you to configure
  which target to connect to  (e.g. ``host`` for an SSH connection or ``device``
  to specific an ADB name) as well as configure other options for the device for
  example the ``working_directory`` or the list of ``modules`` to be loaded onto
  the device.

For more information and a full list of these configuration options please see
:ref:`Run Configuration <run-configuration>`.

Meta Configuration
~~~~~~~~~~~~~~~~~~

The next type of configuration options are the `"Meta Configuration"` options
(for a full list please see :ref:`here <meta-configuration>`) and these are used
to configure the behaviour of WA framework itself, for example directory
locations to be used or logging configuration.


Plugins
~~~~~~~
You can also use this section to supply configuration for specific plugins, such
as augmentations, workloads, resource getters etc. To do this the plugin name
you wish to configure should be provided as an entry in this section and should
contain a mapping of configuration options to their desired settings. If
configuration is supplied for a plugin that is not currently enabled then it will
simply be ignored. This allows for plugins to be temporarily removed
without also having to remove their configuration, or to provide a set of
defaults for a plugin which can then be overridden.

Some plugins provide global aliases which can set one or more configuration
options at once, and these can also bee specified here. For example specifying
the entry ``remote_assets_url`` with a corresponding  value will set the URL the
http resource getter will attempt to search for any missing assets at.


augmentations
"""""""""""""
As mentioned above this section should be used to configure augmentations, both
to specify which should be enabled and disabled and also to provide any relevant
configuration. The ``"augmentation"`` entry, if present, should be a list of
augmentations that should be enabled (or if prefixed with a ``~``, disabled).

.. note:: While augmentations can be enabled and disabled on a per workload
          basis, they cannot yet be re-configured part way through a run and the
          configuration provided as part of an agenda config section or separate
          config file will be used for all jobs in a WA run.

workloads
"""""""""
In addition to configuring individual workloads both in the ``workloads`` and
``sections`` entries you can also provide configuration at this level which will
apply globally in the same style mentioned below. Any configuration provided
here will be overridden if specified again in subsequent sections.


workloads
^^^^^^^^^

Here you can specify a list of workloads to be ran. If you wish to run a
workload with all default values then you can specify the workload name directly
as an entry, otherwise a dict mapping should be provided. Any settings provided
here will be the most specific and therefore override any other more generalised
configuration for that particular workload spec. The valid entries are as
follows:

- :confval:`workload_name` (Mandatory) - The name of the workload to be ran
- :confval:`iterations` - Specify how many iterations the workload should be ran
- :confval:`label` - Similar to IDs but do not have the uniqueness restriction.
  If specified, labels will be used by some output processors instead of (or in
  addition to) the workload name. For example, the csv output processor will put
  the label in the "workload" column of the CSV file.
- :confval:`augmentations` - The instruments and output processors to enable (or
  disabled using a ~) during this workload.
- :confval:`classifiers` Classifiers allow you to tag metrics from this workload
  spec which are often used to help identify what runtime parameters were used
  when post processing results.
- :confval:`workload_parameters` [*workload_params*] - Any parameters to
  configure that particular workload in a dict form.

      .. note:: You can see available parameters for a given workload with the
                :ref:`show command <show-command>`.

- :confval:`runtime_parameters` [*runtime_parms*] - A dict mapping of any
  runtime parameters that should be set for the device for that particular
  workload. For available parameters please see :ref:`runtime parameters
  <runtime-parameters>`.

     .. note:: Unless specified elsewhere these configurations will not be
               undone once the workload has finished. I.e. if the frequency of a
               core is changed it will remain at that frequency until otherwise
               changed.

.. note:: There is also a shorter ``params`` alias available, however this alias will be
          interpreted differently depending on whether it is used in workload
          entry, in which case it will be interpreted as ``workload_params``, or
          at global config or section (see below) level, in which case it will
          be interpreted as ``runtime_params``.


sections
^^^^^^^^

Sections are used for for grouping sets of configuration together in order to
reduce the need for duplicated configuration (for more information please see
:ref:`here <sections>`). Each section specified will be applied for each entry
in the ``workloads`` section. The valid configuration entries are the same
as the ``"workloads"`` section as mentioned above, except you can
additionally specify a "workloads" entry which can be provided with the same
configuration entries as the ``"workloads"`` top level entry.
