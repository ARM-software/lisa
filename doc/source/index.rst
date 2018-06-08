.. Workload Automation 3 documentation master file,

================================================
Welcome to Documentation for Workload Automation
================================================

Workload Automation (WA) is a framework for executing workloads and collecting
measurements on Android and Linux devices. WA includes automation for nearly 30
workloads and supports some common instrumentation (ftrace, hwmon) along with a
number of output formats.

Workload Automation is designed primarily as a developer tool/framework to
facilitate data driven development by providing a method of collecting
measurements from a device in a repeatable way.

Workload Automation is highly extensible. Most of the concrete functionality is
implemented via :ref:`plug-ins <plugin-reference>`, and it is easy to
:ref:`write new plug-ins <writing-plugins>` to support new device types,
workloads, instruments or output processing.

.. contents:: Contents


What's New
==========

.. toctree::
   :maxdepth: 1

   changes
   migration_guide

User Information
================

This section lists general usage documentation. If you're new to WA3, it is
recommended you start with the :doc:`user_guide` page. This section also contains
installation and configuration guides.

.. toctree::
   :maxdepth: 2

   installation
   user_guide
   user_reference


How To Guides
===============

.. toctree::
   :maxdepth: 3

   how_to

FAQ
====

.. toctree::
   :maxdepth: 2

   faq

.. _in-depth:

Developer Information
=====================

This section contains more advanced topics, such how to write your own Plugins
and detailed descriptions of how WA functions under the hood.

.. toctree::
   :maxdepth: 2

   developer_reference


References
==========

.. toctree::
   :maxdepth: 2

   plugins
   glossary

.. Indices and tables
.. ==================

.. .. * :ref:`genindex`
.. .. * :ref:`modindex`
.. * :ref:`search`
