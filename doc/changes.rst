**************
APIs stability
**************
.. _api-stability:

APIs inside LISA are split between private and public ones:

  * Public APIs can be expected to stay stable, or undergo a deprecation cycle
    where they will trigger an :exc:`DeprecationWarning` and be documented as
    such before being removed. Exceptions to that rule are documented explicitly
    as such.

  * Private APIs can be changed at all points.

Public APIs consist of classes and functions with names not starting with an
underscore, defined in modules with a name not starting with an underscore (or
any of its parent modules or containing class).

Everything else is private.

.. note:: User subclassing is usually more at risk of breakage than other uses
    of the APIs. Behaviors are usually not restricted to a single method, which
    means the subclass would have to override multiple of them to preserve
    important API laws. This is unfortunately not future-proof, as new versions
    can add new methods that would also require being overridden and kept in
    sync. If for some reason subclassing is required, please get in touch in the
    `github issue tracker <https://github.com/ARM-software/lisa/issues>`_
    before relying on that for production.

.. note:: Instance attributes are considered public following the same
    convention as functions and classes. Only reading from them is expected in
    user code though, any attempt to modify or delete them is outside of the
    bounds of what the public API exposes (unless stated explicitly otherwise).
    This means that a minor version change could swap an instance attribute for
    a read-only property. It also means that any problem following the
    modification of an attribute by a user will not be considered as a bug.

**********
Versioning
**********

LISA releases on :ref:`PyPI<setup-pypi>` are done following semantic versioning
as defined in https://semver.org/. As pointed by `api-stability`_, classes are
split on the following axes for the purpose of semver tracking:

  * A set of methods and attributes in general: Adding a method entails a minor
    version bump, even though it can technically cause a breaking change in a
    user subclass that happened to use the same name.

  * Inheritance tree: the MRO of a class is not considered as part of the stable
    public API and can therefore change at any point. Classes named ``*Base``
    can usually be relied on for ``issubclass()`` and ``isinstance()`` but that
    is not a hard rule. The reason behind that is that even adding a class to
    the hierarchy can break existing uses of ``isinstance()`` so there is
    essentially no way of making any change to the inheritance tree that is not
    a breaking change.

*********
Changelog
*********

.. exec::

    from lisa.utils import LISA_HOME
    from lisa._doc.helpers import make_changelog

    repo = LISA_HOME
    changelog = make_changelog(
        repo=LISA_HOME,
    )
    print(changelog)

****************
Breaking changes
****************

Here is a list of commits introducing breaking changes in LISA:

.. exec::

    from lisa.utils import LISA_HOME
    from lisa._git import find_commits, log

    pattern = 'BREAK'

    repo = LISA_HOME
    commits = find_commits(repo, grep=pattern)
    ignored_sha1s = {
        '30d75656c7ff8a159dd52164269e69eed6dfccad',
    }
    for sha1 in commits:
        if sha1 in ignored_sha1s:
            continue
        commit_log = log(repo, ref=sha1, format='%cd%n%H%n%B')
        entry = '.. code-block:: text\n\n  {}\n'.format(commit_log.replace('\n', '\n  '))
        print(entry)

***************
Deprecated APIs
***************

Here is a list of deprecated APIs in LISA, sorted by version in which they will
be removed:

.. exec::

    from lisa._doc.helpers import get_deprecated_table
    print(get_deprecated_table())


***************
Release Process
***************

Making a new release involves the following steps:

  1. Update ``version_tuple`` in :mod:`lisa.version`.

  2. Ensure LISA as a whole refers to relevant versions of:

     * Alpine Linux in :mod:`lisa._kmod`
     * Ubuntu in ``Vagrantfile``
     * Binary dependencies in :mod:`lisa._assets`
     * Android SDK installed by ``install_base.sh``
     * Java version used by Android SDK in ``install_base.sh``

  3. Create a ``vX.Y.Z`` tag.

  4. Make the Python wheel. See ``tools/make-release.sh`` for some
     indications on that part.

  5. Install that wheel in a _fresh_ :ref:`Vagrant VM<setup-vagrant>`. Ensure
     that the VM is reinstalled from scratch and that the vagrant box in use is
     up to date.

  6. Run ``tools/tests.sh`` in the VM and ensure no deprecated item scheduled
     for removal in the new version is still present in the sources (should
     result in import-time exceptions).

  7. Ensure all CIs in use are happy.

  8. Push the ``vX.Y.Z`` tag in the main repo

  9. Update the ``release`` branch to be at the same commit as the ``vX.Y.Z`` tag.

  10. Upload the wheel on PyPI.


******************************
Transitioning from LISA legacy
******************************

A big refactoring effort was started in mid 2018, which produced a lot of
(much needed) changes. If you are used to using LISA before the refactoring came
into place, this guide is for you.

Global changes
==============

Project structure
+++++++++++++++++

* ``$repo/libs/utils`` is now ``$repo/lisa/``. ``$repo/libs/wlgen`` has also been
  moved to that location.
* :mod:`devlib` and :mod:`wa` are now under ``$repo/external/``. Git subtrees
  are now used instead of submodules.
* All non self-tests have been moved from ``$repo/tests`` to ``$repo/lisa/tests``

Updating your tree
++++++++++++++++++

Since we no longer use submodules, a ``git pull`` is all you need
(``lisa-update`` no longer exists). Also, see :ref:`kernel-testing-page`.

Python 3
++++++++

With Python 2 end of life drawing near, we decided to bridge the gap and move
over to Python 3. Unlike :mod:`devlib`, we didn't go for compatibility with
both Python 2 and Python 3 - LISA is now Python 3 only.

Imports
+++++++

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
  that could lead to an older LISA/devlib being imported.

Logging
+++++++

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
+++++++++

The LISA shell command to start notebooks has been changed from ``lisa-ipython`` to
``lisa-jupyter`` (the actual notebooks have been Jupyter for several years now).

We also use the newer Jupyterlab, as the regular Jupyter notebooks will slowly
be phased out - see the
`Official Jupyter roadmap <https://github.com/jupyter/roadmap/blob/master/notebook.md>`_.

.. warning::

  Jupyterlab breaks the TRAPpy plots that use JS injection (e.g.
  :class:`~trappy.plotter.ILinePlot`). You can use the "old" notebooks by clicking
  ``Help->Launch Classic Notebook``, but that is bound to go away eventually.

Furthermore, in LISA legacy notebooks served as documentations and where the
main source of examples. We now have a proper documentation (you're reading it!),
so we greatly trimmed down the number of notebooks we had.

We've kept older notebooks in ``ipynb/deprecated``, but they have not been ported
over to the new APIs (or even to Python3) so they won't work. They are there in
case we find a reason to bring back some of them.

API Changes
===========

TestEnv
+++++++

Creating a ``env.TestEnv`` used to look like this::

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

The equivalent class to use is now :class:`lisa.target.Target`. It does not
require a mapping to be built anymore.

We now have a dedicated class for the ``target_conf``, see :class:`lisa.target.TargetConf`.
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

  :class:`~lisa.target.Target` instances can now be easily be created
  :meth:`from the configuration file<lisa.target.Target.from_default_conf>` or
  :meth:`via the CLI<lisa.target.Target.from_cli>`.


Trace
+++++

The :class:`lisa.trace.Trace` class hasn't changed much in terms of functionality,
but we did rename/move things to make them more coherent.

* Removed last occurences of camelCase
* Removed big.LITTLE assumptions and made the code only rely on CPU capacities or
  frequency domains, where relevant.
* Constructor now only takes trace files as input, not folders anymore.
* ``Trace.data_frame`` is gone:

**LISA legacy**::

  trace.data_frame.trace_event("sched_switch")
  # or
  trace.df("sched_switch")

**LISA next**::

  trace.df_event("sched_switch")


Analysis
++++++++

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

  trace.ana.frequency.df_cpu_frequency_transitions(0)

To make autocompletion more useful, all methods returning a :class:`pandas.DataFrame`
will start with ``df_``, whereas all methods rendering a plot will start with ``plot_``.

.. admonition:: Cool new feature

  Trace events required by the analysis methods are now automatically documented,
  see :meth:`~lisa.analysis.frequency.FrequencyAnalysis.df_cpu_frequency_residency`
  for instance.

wlgen
+++++

The :class:`lisa.wlgen.rta.RTA` class has been simplified somewhat:

* :class:`lisa.wlgen.rta.RTATask` no longer has a superfluous ``get()`` method
* There is no longer a split between task and phases.
  :class:`lisa.wlgen.rta.RTAPhase` can be arranged into a tree with arbitrary
  depth, instead of the previous split of toplevel class
  :class:`~lisa.wlgen.rta.RTATask` and :class:`~lisa.wlgen.rta.Phase`.
* ``RTA.conf()`` has been squashed inside alternative constructors, see
  :meth:`lisa.wlgen.rta.RTA.from_str` and :meth:`lisa.wlgen.rta.RTA.from_profile`.
* It is now possible to create a full JSON file without a live target using
  :class:`~lisa.wlgen.rta.RTAConf`.

**LISA legacy**::

  profile = {}
  profile["my_task"] = Periodic(duty_cycle_pct=30).get()

  wload = RTA(te.target, "foo", calibration)
  wload.conf(kind='profile', params=profile)

**LISA next**::

  profile = {
      'my_task': RTAPhase(
          prop_wload=PeriodicWload(
              duty_cycle_pct=30,
              period=16e-3,
              duration=1,
          )
      )
  }

  wload = RTA.from_profile(te, "foo", profile, res_dir, calibration)

Kernel tests
++++++++++++

The ``Executor`` from LISA legacy has been entirely removed, and a new test
framework has been put in place. Tests are now coded as pure Python classes,
which means they can be imported and executed in scripts/notebooks without any
additionnal effort. See :ref:`kernel-testing-page` for more details about
using/writing tests.


Energy Meter
++++++++++++

Energy meters are all subclasses of :class:`lisa.energy_meter.EnergyMeter`.
They can now be created in two ways. For :class:`lisa.energy_meter.HWMon`, this
would give::

  target = Target.from_default_conf()
  res_dir = "/foo/bar"

  # Directly build an instance
  emeter = HWMon(target, channel_map=..., res_dir=res_dir)

  # Or using a configuration file
  conf = HWMonConf.from_yaml_map('path/to/hwmon_conf.yml')
  emeter = HWMon.from_conf(target, conf, res_dir)

with ``hwmon_conf.yml`` containing:

.. code-block:: YAML

  hwmon-conf:
       channel-map: ...

All subclasses of :class:`lisa.energy_meter.EnergyMeter` have a configuration
class named `*Conf`.
