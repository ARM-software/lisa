************
User's guide
************

Installation
============

Native installation
+++++++++++++++++++

For now, you will need to clone LISA from `github <https://github.com/ARM-software/lisa>`_ ,
and then issue these commands::

  git clone https://github.com/ARM-software/lisa.git -b next
  cd lisa
  # A few packages need to be installed, like python3 or kernelshark. Python
  # modules will be installed in a venv at the next step, without touching
  # any system-wide install location.
  sudo ./install_base_ubuntu.sh
  # On the first run, it will take care of creating a Python venv and populating it
  source init_env

In case the venv becomes unusable for some reason, the ``lisa-install``
shell command available after sourcing ``init_env`` will allow to create a new
clean venv from scratch.

Additional Python packages
--------------------------

``lisa-install`` will also install the content of
``$LISA_HOME/custom_requirements.txt`` if the file exists. That allows
re-installing a custom set of packages automatically when the venv needs to
regenerated.

Without automatic ``venv``
--------------------------

Sometimes, LISA needs to operate in an environment setup for multiple tools. In
that case, it may be easier to manage manually a venv/virtualenv instead of
letting LISA create one for its shell.

Setting ``export LISA_USE_VENV=0`` prior to ``source init_env`` will avoid the
creation and usage of the LISA-managed venv. ``lisa-install`` command can still
be used to install the necessary Python packages, which will honor any
venv-like system manually setup.

Alternatively, ``lisa`` package is packaged according to the usual Python
practices, which includes a ``setup.py`` script, and a
``devmode_requirements.txt`` file that will install all the shipped packages in
editable mode (including those that are not developped in that repository, but
still included for convenience).

Virtual machine installation
++++++++++++++++++++++++++++++++++

LISA provides a Vagrant recipe which automates the generation of a
VirtualBox based virtual machine pre-configured to run LISA. To generate and
use such a virtual machine you need:

- `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`__
- `Vagrant <https://www.vagrantup.com/downloads.html>`__

Once these two components are available on your machine, issue these commands::

  git clone https://github.com/ARM-software/lisa.git -b next
  cd lisa
  vagrant up

This last command builds and executes the VM according to the description provided
by the Vagrant file available in the root folder of the LISA source tree.

Once the VM installation is complete, you can access that VM with::

  vagrant ssh

LISA shell
==========

You should now have all of the required dependencies installed. From now on, you
can use the LISA shell, which provides a convenient set of commands for easy
access to many LISA related functions (done automatically by Vagrant)::

  source init_env

.. tip:: Run ``lisa-help`` to see an overview of the provided LISA commands.

Updating
========

Over time, we might change/add some dependencies to LISA. As such, if you
update your LISA repository, you should make sure your locally-installed
packages still match those dependencies. Sourcing ``init_env`` from a
new shell should suffice, which will hint the user if running ``lisa-install``
again is needed.

.. tip::

  A git **post-checkout** hook is provided in ``tools/post-checkout``. It will
  check that no ``setup.py`` file have been updated since last time
  ``lisa-install`` was executed. If a modification is detected, it will ask the
  user to run ``lisa-install`` again, since a dependency might have been added,
  or a version requirement might have been updated.

Notebooks
=========

The LISA shell can simplify starting an Jupyter notebook server::

  [LISAShell lisa] \> lisa-jupyter start

  Starting Jupyter Notebooks...

  Notebook server configuration:
    URL        :  http://127.0.0.1:8888/?token=b34F8D0e457BDa570C4A6D7AF113CB45d9CcAF44Aa7Cf400
    Folder     :  /data/work/lisa/ipynb
    Logfile    :  /data/work/lisa/ipynb/server.log
    PYTHONPATH :
	  /data/work/lisa/modules_root/


  Notebook server task: [4] 30177

Note that the ``lisa-jupyter`` command allows you to specify interface and
port in case you have several network interfaces on your host::

  lisa-jupyter start [interface [port]]

The URL of the main folder served by the server is printed on the screen.
By default it is http://127.0.0.1:8888/. Once the server is started you can
have a look at the provided tutorial notebooks are accessible by following this
`link <http://127.0.0.1:8888/notebooks/examples/typical_experiment.ipynb>`__.
This initial tutorial can be seen (but not executed) also on `github
<https://github.com/ARM-software/lisa/blob/next/ipynb/examples/typical_experiment.ipynb>`__.


Transitioning from LISA legacy
==============================

A big refactoring effort was started in mid 2018, which produced a lot of
(much needed) changes. If you are used to using LISA before the refactoring came
into place, this guide is for you.

Global changes
++++++++++++++

Project structure
-----------------

* ``$repo/libs/utils`` is now ``$repo/lisa/``. ``$repo/libs/wlgen`` has also been
  moved to that location.
* :mod:`devlib`, :mod:`trappy`, :mod:`bart` and :mod:`wa` are
  now under ``$repo/external/``. Git subtrees are now used instead of submodules.
* All non self-tests have been moved from ``$repo/tests`` to ``$repo/lisa/tests``

Updating your tree
------------------

Since we no longer use submodules, a ``git pull`` is all you need
(``lisa-update`` no longer exists). Also, see `Updating`_.

Python 3
--------

With Python 2 end of life drawing near, we decided to bridge the gap and move over
to Python 3. Unlike :mod:`devlib` or :mod:`trappy`, we didn't go for compatibility
with both Python 2 and Python 3 - LISA is now Python 3 only.

Imports
-------

LISA legacy used implicit relative imports with a bit of dark magic to hold
everything together. Say you want to import the :class:`lisa.trace.Trace` class
found in ``lisa/trace.py``, previously you would do it like so::

  from trace import Trace

However, implicit relative imports are dangerous - did you know :class:`trace.Trace`
exists in Python's standard library? This means that with the previous setup, the LISA
module would shadow the standard library's. The above import done in a non-LISA
environment would have imported something completely different!


We now mandate the use of absolute imports, which look like this::

  from lisa.trace import Trace

.. tip::

  This can help you figure out what you are really importing:

    >>> import trace
    >>> print(trace.__path__)
    /usr/lib/python3.5/trace.py

   if that doesn't work you can try

   >>> print(xxx.__file__)

.. warning::

  Do make sure you haven't kept some ``PYTHONPATH`` tweaking in your ``.bashrc``
  that could lead to an older LISA/devlib/trappy being imported.

Logging
-------

Enabling the LISA logger has changed slightly:

**LISA legacy**::

  import logging
  from conf import LisaLogging
  LisaLogging.setup()

**LISA next**::

  import logging
  from lisa.utils import setup_logging
  setup_logging()

Notebooks
---------

The LISA shell command to start notebooks has been changed from ``lisa-ipython`` to
``lisa-jupyter`` (the actual notebooks have been Jupyter for several years now).

We also use the newer Jupyterlab, as the regular Jupyter notebooks will slowly
be phased out - see the
`Official Jupyter roadmap <https://github.com/jupyter/roadmap/blob/master/notebook.md>`_.

.. warning::

  Jupyterlab breaks the TRAPpy plots that use CSS injection (e.g.
  :mod:`~trappy.plotter.ILinePlot`). You can use the "old" notebooks by clicking
  ``Help->Launch Classic Notebook``, but that is bound to go away eventually.

Furthermore, in LISA legacy notebooks served as documentations and where the
main source of examples. We now have a proper documentation (you're reading it!),
so we greatly trimmed down the number of notebooks we had.

We've kept older notebooks in ``ipynb/deprecated``, but they have not been ported
over to the new APIs (or even to Python3) so they won't work. They are there in
case we find a reason to bring back some of them.

API Changes
+++++++++++

TestEnv
-------

Creating a :class:`lisa.env.TestEnv` used to look like this::

    target_conf = {
	# Define the kind of target platform to use for the experiments
	"platform"    : 'linux',

	# Preload settings for a specific target
	"board"       : 'juno',  # juno - JUNO board with mainline hwmon

	# Define devlib module to load
	"modules"     : [
	    'bl',           # enable big.LITTLE support
	    'cpufreq'       # enable CPUFreq support
	],

	"host"        : '192.168.0.1',
	"username"    : 'root',
	"password"    : 'root',

	"rtapp-calib" : {
	    '0': 361, '1': 138, '2': 138, '3': 352, '4': 360, '5': 353
	}
    }

    te = TestEnv(target_conf)

We now have a dedicated class for the ``target_conf``, see :class:`lisa.env.TargetConf`.
The most notable changes are as follows (see the doc for details):

* ``"platform"`` is now ``"kind"``
* ``"board"`` used to load some target-specific settings, which we got rid of.
  The closest thing to it is ``"name"`` which is just a pretty-printing name and
  has no extra impact.
* You don't have to specify devlib modules to load anymore. All (loadable)
  modules are now loaded. If you find some module too slow to load, you can
  specify a list of modules to exclude.
* LISA used to have ``target.config`` JSON file at its root. Its equivalent is
  now ``target_conf.yml``, which is in YAML.

.. admonition:: Cool new feature

  :class:`~lisa.env.TestEnv` instances can now be easily be created
  :meth:`from the configuration file<lisa.env.TestEnv.from_default_conf>` or
  :meth:`via the CLI<lisa.env.TestEnv.from_cli>`.


Trace
-----

The :class:`lisa.trace.Trace` class hasn't changed much in terms of functionality,
but we did rename/move things to make them more coherent.

* Removed last occurences of camelCase
* Removed big.LITTLE assumptions and made the code only rely on CPU capacities or
  frequency domains, where relevant.
* ``Trace.data_frame`` is gone:

**LISA legacy**::

  trace.data_frame.trace_event("sched_switch")
  # or
  trace.df("sched_switch")

**LISA next**::

  trace.df_events("sched_switch")


Analysis
--------

Most of the analysis functionality provided by LISA legacy has made its way into
LISA next, although several functionalities were restructured and merged together.
Most methods were moved into different modules as well in an attempt to instore
some sense of logic - for instance, ``analysis.latency.df_latency`` is now
:meth:`~lisa.analysis.tasks.TasksAnalysis.df_task_states`. An exact changelog would
fill up your screen, so we recommend having a look at :ref:`analysis-page`.

Note that a new :mod:`lisa.analysis.load_tracking` module has been added to
regroup all load-tracking analysis, and provide wrappers to abstract between our
different load tracking trace event versions (e.g.
:meth:`~lisa.analysis.load_tracking.LoadTrackingAnalysis.df_tasks_signals`)

Analysis function calls must now include their respective module:

**LISA legacy**::

  trace.data_frame.cpu_frequency_transitions(0)

**LISA next**::

  trace.analysis.frequency.df_cpu_frequency_transitions(0)

To make autocompletion more useful, all methods returning a :class:`pandas.DataFrame`
will start with ``df_``, whereas all methods rendering a plot will start with ``plot_``.

.. admonition:: Cool new feature

  Trace events required by the analysis methods are now automatically documented,
  see :meth:`~lisa.analysis.frequency.FrequencyAnalysis.df_cpu_frequency_residency`
  for instance.

wlgen
-----

The :class:`lisa.wlgen.rta.RTA` class has been simplified somewhat:

* :class:`lisa.wlgen.rta.RTATask` no longer has a superfluous ``get()`` method
* ``RTA.conf()`` has been squashed inside alternative constructors, see
  :meth:`lisa.wlgen.rta.RTA.by_str` and :meth:`lisa.wlgen.rta.RTA.by_profile`.

**LISA legacy**::

  profile = {}
  profile["my_task"] = Periodic(duty_cycle_pct=30).get()

  wload = RTA(te.target, "foo", calibration)
  wload.conf(kind='profile', params=profile)

**LISA next**::

  profile = {}
  profile["my_task"] = Periodic(duty_cycle_pct=30)

  wload = RTA.by_profile(te, "foo", profile, res_dir, calibration)

Kernel tests
------------

The ``Executor`` from LISA legacy has been entirely removed, and a new test
framework has been put in place. Tests are now coded as pure Python classes,
which means they can be imported and executed in scripts/notebooks without any
additionnal effort. See :ref:`kernel-testing-page` for more details about
using/writing tests.
