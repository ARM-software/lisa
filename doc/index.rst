.. LISA documentation master file, created by
   sphinx-quickstart on Tue Dec 13 14:20:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: lisa

LISA Documentation
==================

LISA - "Linux Integrated System Analysis" is a toolkit for interactive analysis
and automated regression testing of Linux kernel behaviour.

- See the README on the project's `Github home page`__ for an overview.
- Once you have LISA running, take a look at the tutorial and example notebooks
  included with the installation.

__ https://github.com/ARM-software/lisa

Contributions to LISA and its documentation are very welcome, and handled
via Github pull requests.

.. _Readme:

Contents:

.. toctree::
   :maxdepth: 2

   ────── Getting started ────── <self>

   overview
   setup
   workflows/index

   ──── Tools documentation ──── <self>

   lisa_shell/man/man
   bisector/index
   exekall/index

   ───── API documentation ───── <self>

   target
   workloads
   kernel_tests
   trace_analysis
   energy_analysis
   misc_utilities
   stat_comparison

   ──────── Changes ──────── <self>

   changes
   contributors_guide


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Building this documentation
==============================

- Install ``doc`` optional dependencies of ``lisa`` package (``lisa-install``
  does that by default)
- Run:

  .. code:: shell

    source init_env
    lisa-doc-build

- Find the HTML in ``doc/_build/html``
