.. _configuration-specification:


Configuration
=============

.. include:: user_information/user_reference/agenda.rst

---------------------

.. _run-configuration:

Run Configuration
------------------
In addition to specifying run execution parameters through an agenda, the
behaviour of WA can be modified through configuration file(s). The default
configuration file is ``~/.workload_automation/config.yaml``  (the location can
be changed by setting ``WA_USER_DIRECTORY`` environment variable, see
:ref:`envvars` section below). This file will be created when you first run WA
if it does not already exist. This file must always exist and will always be
loaded. You can add to or override the contents of that file on invocation of
Workload Automation by specifying an additional configuration file using
``--config`` option. Variables with specific names  will be picked up by the
framework and used to modify the behaviour of Workload automation e.g.
the ``iterations`` variable might be specified to tell WA how many times to run
each workload.

---------------------

.. _available_settings:

.. include:: run_config/Run_Configuration.rst

---------------------

.. _meta-configuration:

Meta Configuration
------------------

There are also a couple of settings are used to provide additional metadata
for a run. These may get picked up by instruments or output processors to
attach context to results.

.. include:: run_config/Meta_Configuration.rst

---------------------

.. _envvars:

Environment Variables
---------------------

In addition to standard configuration described above, WA behaviour can be
altered through environment variables. These can determine where WA looks for
various assets when it starts.

:WA_USER_DIRECTORY: This is the location WA will look for config.yaml, plugins,
   dependencies, and it will also be used for local caches, etc. If this
   variable is not set, the default location is ``~/.workload_automation`` (this
   is created when WA is installed).

   .. note:: This location **must** be writable by the user who runs WA.


:WA_LOG_BUFFER_CAPACITY: Specifies the capacity (in log records) for the early
    log handler which is used to buffer log records until a log file becomes
    available. If the is not set, the default value of ``1000`` will be used.
    This should sufficient for most scenarios, however this may need to be
    increased, e.g. if plugin loader scans a very large number of locations;
    this may also be set to a lower value to reduce WA's memory footprint on
    memory-constrained hosts.

---------------------

.. include:: user_information/user_reference/runtime_parameters.rst

---------------------

.. _config-merging:

Configuration Merging
---------------------
WA configuration can come from various sources of increasing priority, as well
as being specified in a generic and specific manner. For example WA's global
config file would be considered the least specific vs the parameters of a
workload in an agenda which would be the most specific. WA has two rules for the
priority of configuration:

    - Configuration from higher priority sources overrides configuration from
      lower priority sources.
    - More specific configuration overrides less specific configuration.

There is a situation where these two rules come into conflict. When a generic
configuration is given in config source of high priority and a specific
configuration is given in a config source of lower priority. In this situation
it is not possible to know the end users intention and WA will error.

This functionality allows for defaults for plugins, targets etc. to be
configured at a global level and then seamless overridden without the need to
remove the high level configuration.

Dependent on specificity, configuration parameters from different sources will
have different inherent priorities. Within an agenda, the configuration in
"workload" entries will be more specific than "sections" entries, which in turn
are more specific than parameters in the "config" entry.

.. _config-include:

Configuration Includes
----------------------

It is possible to include other files in your config files and agendas. This is
done by specifying ``include#`` (note the trailing hash) as a key in one of the
mappings, with the value being the path to the file to be included. The path
must be either absolute, or relative to the location of the file it is being
included from (*not* to the current working directory). The path may also
include ``~`` to indicate current user's home directory.

The include is performed by removing the ``include#`` loading the contents of
the specified into the mapping that contained it. In cases where the mapping
already contains the key to be loaded, values will be merged using the usual
merge method (for overwrites, values in the mapping take precedence over those
from the included files).

Below is an example of an agenda that includes other files. The assumption is
that all of those files are in one directory

.. code-block:: yaml

    # agenda.yaml
    config:
       augmentations: [trace-cmd]
       include#: ./my-config.yaml
    sections:
       - include#: ./section1.yaml
       - include#: ./section2.yaml
    include#: ./workloads.yaml

.. code-block:: yaml

   # my-config.yaml
   augmentations: [cpufreq]


.. code-block:: yaml

   # section1.yaml
   runtime_parameters:
      frequency: max

.. code-block:: yaml

   # section2.yaml
   runtime_parameters:
      frequency: min

.. code-block:: yaml

   # workloads.yaml
   workloads:
      - dhrystone
      - memcpy

The above is equivalent to having a single file like this:

.. code-block:: yaml

    # agenda.yaml
    config:
       augmentations: [cpufreq, trace-cmd]
    sections:
       - runtime_parameters:
            frequency: max
       - runtime_parameters:
            frequency: min
    workloads:
       - dhrystone
       - memcpy

Some additional details about the implementation and its limitations:

- The ``include#`` *must* be a key in a mapping, and the contents of the
  included file *must* be a mapping as well; it is not possible to include a
  list (e.g. in the examples above ``workload:`` part *must* be in the included
  file.
- Being a key in a mapping, there can only be one ``include#`` entry per block.
- The included file *must* have a ``.yaml`` extension.
- Nested inclusions *are* allowed. I.e. included files may themselves include
  files; in such cases the included paths must be relative to *that* file, and
  not the "main" file.

