.. LISA documentation master file, created by
   sphinx-quickstart on Tue Dec 13 14:20:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _bisector-doc:

**********************
Bisector Documentation
**********************

Overview
========

Bisector is a ``git bisect run`` compatible tool used in LISA. Check out the
project's `GitLab`__ for some guides to installation and setup.

``bisector`` allows setting up the steps of a test iteration, repeating
them an infinite number of times (by default). These steps can involve flashing
the board, rebooting it, and running a test command. If one step goes wrong,
bisector implements the logic to retry, abort it, mark it as good or bad
depending on the type of step used. By now, you may have noticed some
similarities between ``bisector`` behaviour and what is expected of the command
executed by ``git bisect run`` [#]_: that is no coincidence, as both ``bisector
run`` and ``bisector report`` can be used as a ``git bisect run``-compliant
script.

``bisector run`` records all the output of the steps in a machine-readable
report that can be inspected using ``bisector report``. The emphasis is put on
reliability against unexpected interruption, flaky commands and other issues
that happen on long running sessions. ``bisector`` will never leave you with an
inconsistent report, or worse, no report at all. A new report is saved after
each iteration and can be inspected as the execution goes on.

__ https://gitlab.arm.com/tooling/lisa
.. [#] https://git-scm.com/docs/git-bisect

Contents
========

.. toctree::
   :maxdepth: 2

   man/man
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



