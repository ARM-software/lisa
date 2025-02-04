.. _agenda-reference:

Agenda
------


An agenda can be thought of as a way to define an experiment as it specifies
what is to be done during a Workload Automation run. This includes which
workloads will be run, with what configuration and which augmentations will be
enabled, etc. Agenda syntax is designed to be both succinct and expressive and
is written using YAML notation.

There are three valid top level entries which are:
:ref:`config <config-agenda-entry>`, :ref:`workloads <workloads-agenda-entry>`,
:ref:`sections <sections-agenda-entry>`.

An example agenda can be seen here:

.. code-block:: yaml

    config:                     # General configuration for the run
        user_directory: ~/.workload_automation/
        default_output_directory: 'wa_output'
        augmentations:          # A list of all augmentations to be enabled and disabled.
        - trace-cmd
        - csv
        - ~dmesg                # Disable the dmseg augmentation

        iterations: 1           # How many iterations to run each workload by default

        device: generic_android
        device_config:
            device: R32C801B8XY # The adb name of our device we want to run on
            disable_selinux: true
            load_default_modules: true
            package_data_directory: /data/data

        trace-cmd:              # Provide config for the trace-cmd augmentation.
            buffer_size_step: 1000
            events:
            - sched*
            - irq*
            - power*
            - thermal*
            no_install: false
            report: true
            report_on_target: false
            mode: write-to-disk
        csv:                    # Provide config for the csv augmentation
            use_all_classifiers: true

    sections:                   # Configure what sections we want and their settings
        - id: LITTLES           # Run workloads just on the LITTLE cores
          runtime_parameters:   # Supply RT parameters to be used for this section
                num_little_cores: 4
                num_big_cores: 0

        - id: BIGS               # Run workloads just on the big cores
          runtime_parameters:    # Supply RT parameters to be used for this section
                num_big_cores: 4
                num_little_cores: 0

    workloads:                  # List which workloads should be run
    -   name: benchmarkpi
        augmentations:
            - ~trace-cmd        # Disable the trace-cmd instrument for this workload
        iterations: 2           # Override the global number of iteration for this workload
        params:                 # Specify workload parameters for this workload
            cleanup_assets: true
            exact_abi: false
            force_install: false
            install_timeout: 300
            markers_enabled: false
            prefer_host_package: true
            strict: false
            uninstall: false
    -   dhrystone               # Run the dhrystone workload with all default config

This agenda will result in a total of 6 jobs being executed on our Android
device, 4 of which running the BenchmarkPi workload with its customized workload
parameters and 2 running dhrystone with its default configuration. The first 3
will be running on only the little cores and the latter running on the big
cores. For all of the jobs executed the output will be processed by the ``csv``
processor,(plus any additional processors enabled in the default config file),
however trace data will only be collected for the dhrystone jobs.

.. _config-agenda-entry:

config
^^^^^^^

This section is used to provide overall configuration for WA and its run. The
``config`` section of an agenda will be merged with any other configuration
files provided (including the default config file) and merged with the most
specific configuration taking precedence (see
:ref:`Config Merging <config-merging>` for more information. The only
restriction is that ``run_name`` can only be specified in the config section
of an agenda as this would not make sense to set as a default.

Within this section there are multiple distinct types of configuration that can
be provided. However in addition to the options listed here all configuration
that is available for :ref:`sections <sections-agenda-entry>` can also be entered
here and will be globally applied.

Configuration
"""""""""""""

The first is to configure the behaviour of WA and how a run as a
whole will behave. The most common options that that you may want to specify are:

  :device: The name of the 'device' that you wish to perform the run
           on. This name is a combination of a devlib
           `Platform <http://devlib.readthedocs.io/en/latest/platform.html>`_ and
           `Target <http://devlib.readthedocs.io/en/latest/target.html>`_. To
           see the available options please use ``wa list targets``.
  :device_config: The is a dict mapping allowing you to configure which target
                  to connect to  (e.g. ``host`` for an SSH connection or
                  ``device`` to specific an ADB name) as well as configure other
                  options for the device for example the ``working_directory``
                  or the list of ``modules`` to be loaded onto the device. (For
                  more information please see
                  :ref:`here <android-general-device-setup>`)
  :execution_order: Defines the order in which the agenda spec will be executed.
  :reboot_policy: Defines when during execution of a run a Device will be rebooted.
  :max_retries: The maximum number of times failed jobs will be retried before giving up.
  :allow_phone_home: Prevent running any workloads that are marked with ‘phones_home’.

For more information and a full list of these configuration options please see
:ref:`Run Configuration <run-configuration>` and
:ref:`Meta Configuration <meta-configuration>`.


Plugins
"""""""
  :augmentations: Specify a list of which augmentations should be enabled (or if
      prefixed with a ``~``, disabled).

      .. note:: While augmentations can be enabled and disabled on a per workload
                basis, they cannot yet be re-configured part way through a run and the
                configuration provided as part of an agenda config section or separate
                config file will be used for all jobs in a WA run.

  :<plugin_name>: You can also use this section to supply configuration for
      specific plugins, such as augmentations, workloads, resource getters etc.
      To do this the plugin name you wish to configure should be provided as an
      entry in this section and should contain a mapping of configuration
      options to their desired settings. If configuration is supplied for a
      plugin that is not currently enabled then it will simply be ignored. This
      allows for plugins to be temporarily removed without also having to remove
      their configuration, or to provide a set of defaults for a plugin which
      can then be overridden.

  :<global_alias>: Some plugins provide global aliases which can set one or more
      configuration options at once, and these can also be specified here. For
      example if you specify a value for the entry ``remote_assets_url`` this
      will set the URL the http resource getter will use when searching for any
      missing assets.

---------------------------

.. _workloads-agenda-entry:

workloads
^^^^^^^^^

Here you can specify a list of workloads to be run. If you wish to run a
workload with all default values then you can specify the workload name directly
as an entry, otherwise a dict mapping should be provided. Any settings provided
here will be the most specific and therefore override any other more generalised
configuration for that particular workload spec. The valid entries are as
follows:

:workload_name: **(Mandatory)** The name of the workload to be run
:iterations: Specify how many iterations the workload should be run
:label: Similar to IDs but do not have the uniqueness restriction.
    If specified, labels will be used by some output processors instead of (or in
    addition to) the workload name. For example, the csv output processor will put
    the label in the "workload" column of the CSV file.
:augmentations: The instruments and output processors to enable (or
    disabled using a ~) during this workload.
:classifiers: Classifiers allow you to tag metrics from this workload
    spec which are often used to help identify what runtime parameters were used
    when post processing results.
:workload_parameters: Any parameters to
    configure that particular workload in a dict form.

    Alias: ``workload_params``

      .. note:: You can see available parameters for a given workload with the
                :ref:`show command <show-command>` or look it up in the
                :ref:`Plugin Reference <plugin-reference>`.

:runtime_parameters: A dict mapping of any runtime parameters that should be set
     for the device for that particular workload. For available
     parameters please see
     :ref:`runtime parameters <runtime-parameters>`.

     Alias: ``runtime_parms``

     .. note:: Unless specified elsewhere these configurations will not be
               undone once the workload has finished. I.e. if the frequency of a
               core is changed it will remain at that frequency until otherwise
               changed.

.. note:: There is also a shorter ``params`` alias available, however this alias will be
          interpreted differently depending on whether it is used in workload
          entry, in which case it will be interpreted as ``workload_params``, or
          at global config or section (see below) level, in which case it will
          be interpreted as ``runtime_params``.


---------------------------

.. _sections-agenda-entry:

sections
^^^^^^^^

Sections are used for for grouping sets of configuration together in order to
reduce the need for duplicated configuration (for more information please see
:ref:`Sections <sections>`). Each section specified will be applied for each
entry in the ``workloads`` section. The valid configuration entries are the
same as the ``"workloads"`` section as mentioned above, except you can
additionally specify:

:workloads: An entry which can be provided with the same configuration entries
    as the :ref:`workloads <workloads-agenda-entry>` top level entry.
