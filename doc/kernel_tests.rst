.. _kernel-testing-page:

**************
Kernel testing
**************

Introduction
============

The LISA kernel tests are mostly meant for regression testing, or for supporting
new submissions to LKML. The existing infrastructure can also be used to hack up
workloads that would allow you to poke specific areas of the kernel.

Tests do not **have** to target Arm platforms nor the task scheduler. The only
real requirement is to be able to abstract your target through
:mod:`devlib`, and from there you are free to implement tests as you see fit.

They are commonly split into two steps:
  1) Collect some data by doing work on the target
  2) Post-process the collected data

In our case, the data usually consists of
`Ftrace <https://www.kernel.org/doc/Documentation/trace/ftrace.txt>`_ traces
that we then parse into :class:`pandas.DataFrame`.

.. seealso:: :ref:`analysis-page`

Available tests
===============

The following tests are available. They can be used as:

  * direct execution using ``lisa-test`` command (``LISA shell``) and ``exekall``
    (see :ref:`automated-testing-page`)
  * the individual classes/methods they are composed of can be used in custom
    scripts/jupyter notebooks (see ipynb/tests/synthetics_example.ipynb)

.. run-command::
  :capture-stderr:

  # Disable warnings to avoid dependencies to break the reStructuredText output
  export PYTHONWARNINGS="ignore"
  exekall run lisa lisa_tests --rst-list --inject-empty-target-conf

Running tests
=============

From the CLI
++++++++++++

The shortest path to executing a test from a shell is:

1. Update the ``target_conf.yml`` file located at the root of the repository
   with the credentials to connect to the development board (see
   :class:`~lisa.target.TargetConf` keys for more information)
2. Run the following:

  .. code-block:: sh

    # To run all tests
    lisa-test
    # To list available tests
    lisa-test --list
    # To run a test matching a pattern
    lisa-test '*test_task_placement'

    # For reStructuredText to have its bullet point closing blank line
    printf "\n"


More advanced workflows are described at :ref:`automated-testing-page`.

From a python environment
+++++++++++++++++++++++++

See the usage example of :class:`~lisa.tests.base.TestBundle`

Writing tests
=============

Concepts
++++++++

Writing scheduler tests can be tricky, especially when you're
trying to make them work without relying on custom tracepoints (which is
what you should aim for). Sometimes, a good chunk of the test code will be
about trying to get the target in an expected initial state, or preventing some
undesired mechanic from barging in. That's why we rely on the freezer cgroup to
reduce the amount of noise introduced by the userspace, but it's not solving all
of the issues. As such, your tests should be designed to:

a. minimize the amount of non test-related noise (e.g. freezer)
b. withstand events we can't control (use error margins, averages...)

The main class of the kernel tests is :class:`~lisa.tests.base.TestBundle`.
Have a look at its documentation for implementation and usage examples.

The relationship between the test classes has been condensed into this diagram,
although you'll find more details in the API documentation of these classes.

.. uml::

  class TestMetric {
	+ data
	+ units
  }


  note bottom of TestMetric {
       TestMetrics serve to answer
       <b>"Why did my test fail/pass ?"</b>.
       They are free-form, so they can be
       error counts, durations, stats...
  }

  class Result {
	PASSED
	FAILED
	UNDECIDED
  }

  class ResultBundle {
	+ result : Result
	+ add_metric()
  }

  ResultBundle "1" *- "1" Result
  ' This forces a longer arrow ------------v
  ResultBundle "1" *- "1..*" TestMetric : "          "

  class TestBundleBase {
      # _from_target() : TestBundleBase
      + from_target() : TestBundleBase
      + from_dir() : TestBundleBase
  }

  note right of TestBundleBase {
      Methods returning <b>TestBundleBase</b>
      are alternative constructors

      <b>from_target()</b> does some generic
      work, then calls <b>_from_target()</b>. You'll
      have to override it depending on what
      you want to execute on the target.
  }

  class MyTestBundle {
	# _from_target() : TestBundleBase
	+ test_foo_is_bar() : ResultBundle
  }

  note right of MyTestBundle {
      Non-abstract <b>TestBundleBase</b> classes
      must define test methods that return
      a <b>ResultBundle</b>
  }

  TestBundleBase <|-- MyTestBundle
  MyTestBundle .. ResultBundle

Implementations of :class:`~lisa.tests.base.TestBundleBase._from_target` can
execute any sort of arbitry Python code. This means that you are free to
manipulate sysfs entries, or to execute arbitray binaries on the target. The
:class:`~lisa.wlgen.workload.Workload` class has been created to
facilitate the execution of commands/binaries on the target.

An important daughter class of :class:`~lisa.wlgen.workload.Workload`
is :class:`~lisa.wlgen.rta.RTA`, as it facilitates the creation and
execution of `rt-app <https://github.com/scheduler-tools/rt-app>`_ workloads.
It is very useful for scheduler-related tests, as it makes it easy to create
tasks with a pre-determined utilization.

Example
+++++++

Here is a commented example of an ``rt-app``-based test, showcasing the APIs
that are commonly used to write such tests.

It can be executed using:

.. code-block:: sh

    exekall run lisa lisa_tests.test_example --conf $LISA_CONF

.. exec::
    # Check that links inside 'test_example.py' are not broken.
    from lisa._doc.helpers import check_dead_links
    from lisa_tests import test_example
    check_dead_links(test_example.__file__)

.. literalinclude:: ../lisa_tests/test_example.py
   :language: python
   :pyobject: ExampleTestBundle
   :linenos:

API
===

Base classes
++++++++++++

.. automodule:: lisa.tests.base
   :members:
