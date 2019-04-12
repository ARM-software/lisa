.. LISA documentation master file, created by
   sphinx-quickstart on Tue Dec 13 14:20:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LISA API Documentation
==============================================

LISA - "Linux Integrated System Analysis" is a toolkit for interactive analysis
and automated regression testing of Linux kernel behaviour.

- See the README on the project's `Github home page`__ for an overview.
- Check out the project's `Github Wiki`__ for some guides to installation
  and setup.
- Once you have LISA running, take a look at the tutorial and example notebooks
  included with the installation.

__ https://github.com/ARM-software/lisa
__ https://github.com/ARM-software/lisa/wiki

Contributions to LISA and its documentation are very welcome, and handled
via Github pull requests.

.. _Readme:

Contents:

.. toctree::
   :maxdepth: 2

   first_separator

   overview
   setup
   lisa_shell

   guides_separator

   transition_guide
   contributors_guide
   workflows/index

   api_separator

   target
   workloads
   kernel_tests
   trace_analysis
   energy_analysis
   misc_utilities

   tools_separator

   bisector
   exekall

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Building this documentation
==============================
- Install ``sphinx-doc``
- From the root of the LISA source tree: ``cd doc && make html``
- Find the HTML in ``doc/_build/html``
