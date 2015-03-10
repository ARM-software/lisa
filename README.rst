Workload Automation
+++++++++++++++++++

Workload Automation (WA) is a framework for executing workloads and collecting
measurements on Android and Linux devices. WA includes automation for nearly 50
workloads (mostly Android), some common instrumentation (ftrace, ARM
Streamline, hwmon).  A number of output formats are supported. 

Workload Automation is designed primarily as a developer tool/framework to
facilitate data driven development by providing a method of collecting
measurements from a device in a repeatable way.

Workload Automation is highly extensible. Most of the concrete functionality is
implemented via plug-ins, and  it is easy to write new plug-ins to support new
device types, workloads, instrumentation or output processing. 


Requirements
============

- Python 2.7
- Linux (should work on other Unixes, but untested)
- Latest Android SDK (ANDROID_HOME must be set) for Android devices, or
- SSH for Linux devices


Installation
============

To install::

        python setup.py sdist
        sudo pip install dist/wlauto-*.tar.gz

Please refer to the `installation section <./doc/source/installation.rst>`_ 
in the documentation for more details.


Basic Usage
===========

Please see the `Quickstart <./doc/source/quickstart.rst>`_ section of the 
documentation.


Documentation
=============

Documentation in reStructuredText format may be found under ``doc/source``. To
compile it into cross-linked HTML, make sure you have `Sphinx
<http://sphinx-doc.org/install.html>`_ installed, and then ::

        cd doc
        make html


License
=======

Workload Automation is distributed under `Apache v2.0 License
<http://www.apache.org/licenses/LICENSE-2.0>`_. Workload automation includes
binaries distributed under differnt licenses (see LICENSE files in specfic
directories).


Feedback, Contrubutions and Support
===================================

- Please use the GitHub Issue Tracker associated with this repository for
  feedback.
- ARM licensees may contact ARM directly via their partner managers.
- We welcome code contributions via GitHub Pull requests. Please see
  "Contributing Code" section of the documentation for details.
