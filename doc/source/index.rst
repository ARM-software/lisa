.. Workload Automation 3 documentation master file,

================================================
Welcome to Documentation for Workload Automation
================================================

Workload Automation (WA) is a framework for executing workloads and collecting
measurements on Android and Linux devices. WA includes automation for nearly 40
workloads and supports some common instrumentation (ftrace, hwmon) along with a
number of output formats.

WA is designed primarily as a developer tool/framework to facilitate data driven
development by providing a method of collecting measurements from a device in a
repeatable way.

WA is highly extensible. Most of the concrete functionality is
implemented via :ref:`plug-ins <plugin-reference>`, and it is easy to
:ref:`write new plug-ins <writing-plugins>` to support new device types,
workloads, instruments or output processing.

.. note:: To see the documentation of individual plugins please see the
          :ref:`Plugin Reference <plugin-reference>`.

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
recommended you start with the :ref:`User Guide <user-guide>` page. This section also contains
installation and configuration guides.

.. toctree::
   :maxdepth: 3

   user_information


.. _in-depth:

Developer Information
=====================

This section contains more advanced topics, such how to write your own Plugins
and detailed descriptions of how WA functions under the hood.

.. toctree::
   :maxdepth: 3

   developer_information


Plugin Reference
================

.. toctree::
   :maxdepth: 2

   plugins

API
===

.. toctree::
    :maxdepth: 2

    api

Glossary
========

.. toctree::
    :maxdepth: 2

    glossary

FAQ
====

.. toctree::
   :maxdepth: 2

   faq
