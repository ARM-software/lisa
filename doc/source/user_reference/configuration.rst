.. _configuration-specification:


Configuration
=============

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
framework and used to modify the behaviour of Workload automation.

.. _available_settings:

.. include:: run_config/Run_Configuration.rst

Meta Configuration
------------------

There are also a couple of settings are used to provide additional metadata
for a run. These may get picked up by instruments or output processors to
attach context to results.

.. include:: run_config/Meta_Configuration.rst


.. _envvars:

Environment Variables
---------------------

In addition to standard configuration described above, WA behaviour can be
altered through environment variables. These can determine where WA looks for
various assets when it starts.

.. confval:: WA_USER_DIRECTORY

   This is the location WA will look for config.yaml, plugins,  dependencies,
   and it will also be used for local caches, etc. If this variable is not set,
   the default location is ``~/.workload_automation`` (this is created when WA
   is installed).

   .. note:: This location **must** be writable by the user who runs WA.


.. include:: user_reference/runtime_parameters.rst

