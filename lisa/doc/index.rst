.. LISA documentation master file, created by
   sphinx-quickstart on Tue Dec 13 14:20:00 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LISA API Documentation
==============================================

LISA - "Linux Integrated System Analysis" is a toolkit for interactive analysis
and automated regression testing of Linux kernel behaviour. To get started with LISA:

- See the README on the project's `Github home page`__ for an overview.
- Check out the project's `Github Wiki`__ for some guides to installation
  and setup.
- Once you have LISA running, take a look at the tutorial and example notebooks
  included with the installation.

__ https://github.com/ARM-software/lisa
__ https://github.com/ARM-software/lisa/wiki

This site contains documentation for LISA's APIs. For some parts of LISA, API
documentation is a work-in-progress. Where the API documentation is lacking, see
the example/tutorial notebooks provided with LISA, or just dive in and read the
code. Contributions to LISA and its documentation are very welcome, and handled
via Github pull requests.

.. _Readme:

Contents:

.. TODO: due to our slightly weird package structure the index here is wildly
   nested where it needn't be.

.. TODO: Move wiki to here, wirte a proper module doc, proove Riemann's hypothesis

.. toctree::
   :maxdepth: 2

   tests
   wlgen
   internals


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
