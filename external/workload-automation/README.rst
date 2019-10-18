Workload Automation
+++++++++++++++++++

Workload Automation (WA) is a framework for executing workloads and collecting
measurements on Android and Linux devices. WA includes automation for nearly 40
workloads and supports some common instrumentation (ftrace, hwmon) along with a
number of output formats.

WA is designed primarily as a developer tool/framework to facilitate data driven
development by providing a method of collecting measurements from a device in a
repeatable way.

WA is highly extensible. Most of the concrete functionality is implemented via
plug-ins, and it is easy to write new plug-ins to support new device types,
workloads, instruments or output processing.


Requirements
============

- Python 3
- Linux (should work on other Unixes, but untested)
- Latest Android SDK (ANDROID_HOME must be set) for Android devices, or
- SSH for Linux devices


Installation
============

To install::

        git clone git@github.com:ARM-software/workload-automation.git workload-automation
        sudo -H python setup [install|develop]

Note: A `requirements.txt` is included however this is designed to be used as a
reference for known working versions rather than as part of a standard
installation.

Please refer to the `installation section <http://workload-automation.readthedocs.io/en/latest/user_information.html#install>`_
in the documentation for more details.


Basic Usage
===========

Please see the `Quickstart <http://workload-automation.readthedocs.io/en/latest/user_information.html#user-guide>`_
section of the documentation.


Documentation
=============

You can view pre-built HTML documentation `here <http://workload-automation.readthedocs.io/en/latest/>`_.

Documentation in reStructuredText format may be found under ``doc/source``. To
compile it into cross-linked HTML, make sure you have `Sphinx
<http://sphinx-doc.org/install.html>`_ installed, and then ::

        cd doc
        make html


License
=======

Workload Automation is distributed under `Apache v2.0 License
<http://www.apache.org/licenses/LICENSE-2.0>`_. Workload automation includes
binaries distributed under different licenses (see LICENSE files in specific
directories).


Feedback, Contributions and Support
===================================

- Please use the GitHub Issue Tracker associated with this repository for
  feedback.
- ARM licensees may contact ARM directly via their partner managers.
- We welcome code contributions via GitHub Pull requests. Please see
  "Contributing Code" section of the documentation for details.
