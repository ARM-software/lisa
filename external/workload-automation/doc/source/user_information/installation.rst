.. _installation:

************
Installation
************

.. contents:: Contents
   :depth: 2
   :local:


.. module:: wa

This page describes the 3 methods of installing Workload Automation 3. The first
option is to use :ref:`pip` which will install the latest release of WA, the
latest development version from :ref:`github <github>` or via a
:ref:`dockerfile`.


Prerequisites
=============

Operating System
----------------

WA runs on a native Linux install. It was tested with Ubuntu 14.04,
but any recent Linux distribution should work. It should run on either
32-bit or 64-bit OS, provided the correct version of Android (see below)
was installed. Officially, **other environments are not supported**. WA
has been known to run on Linux Virtual machines and in Cygwin environments,
though additional configuration may be required in both cases (known issues
include makings sure USB/serial connections are passed to the VM, and wrong
python/pip binaries being picked up in Cygwin). WA *should* work on other
Unix-based systems such as BSD or Mac OS X, but it has not been tested
in those environments. WA *does not* run on Windows (though it should be
possible to get limited functionality with minimal porting effort).

.. Note:: If you plan to run Workload Automation on Linux devices only,
          SSH is required, and Android SDK is optional if you wish
          to run WA on Android devices at a later time. Then follow the
          steps to install the necessary python packages to set up WA.

          However, you would be starting off with a limited number of
          workloads that will run on Linux devices.

Android SDK
-----------

You need to have the Android SDK with at least one platform installed.
To install it, download the ADT Bundle from here_.  Extract it
and add ``<path_to_android_sdk>/sdk/platform-tools`` and ``<path_to_android_sdk>/sdk/tools``
to your ``PATH``.  To test that you've installed it properly, run ``adb
version``. The output should be similar to this::

        adb version
        Android Debug Bridge version 1.0.39

.. _here: https://developer.android.com/sdk/index.html

Once that is working, run ::

        android update sdk

This will open up a dialog box listing available android platforms and
corresponding API levels, e.g. ``Android 4.3 (API 18)``. For WA, you will need
at least API level 18 (i.e. Android 4.3), though installing the latest is
usually the best bet.

Optionally (but recommended), you should also set ``ANDROID_HOME`` to point to
the install location of the SDK (i.e. ``<path_to_android_sdk>/sdk``).


Python
------

Workload Automation 3 currently supports both Python 2.7 and Python 3.

.. _pip:

pip
---

pip is the recommended package manager for Python. It is not part of standard
Python distribution and would need to be installed separately. On Ubuntu and
similar distributions, this may be done with APT::

        sudo apt-get install python-pip

.. note:: Some versions of pip (in particluar v1.5.4 which comes with Ubuntu
          14.04) are know to set the wrong permissions when installing
          packages, resulting in WA failing to import them. To avoid this it
          is recommended that you update pip and setuptools before proceeding
          with installation::

                  sudo -H pip install --upgrade pip
                  sudo -H pip install --upgrade setuptools

          If you do run  into this issue after already installing some packages,
          you can resolve it by running ::

                  sudo chmod -R a+r /usr/local/lib/python2.7/dist-packages
                  sudo find /usr/local/lib/python2.7/dist-packages -type d -exec chmod a+x {} \;

          (The paths above will work for Ubuntu; they may need to be adjusted
          for other distros).


Python Packages
---------------

.. note:: pip should automatically download and install missing dependencies,
          so if you're using pip, you can skip this section. However some
          packages the will be installed have C plugins and will require Python
          development headers to install. You can get those by installing
          ``python-dev`` package in apt on Ubuntu (or the equivalent for your
          distribution).

Workload Automation 3 depends on the following additional libraries:

  * pexpect
  * docutils
  * pySerial
  * pyYAML
  * python-dateutil
  * louie
  * pandas
  * devlib
  * wrapt
  * requests
  * colorama
  * future

You can install these with pip::

        sudo -H pip install pexpect
        sudo -H pip install pyserial
        sudo -H pip install pyyaml
        sudo -H pip install docutils
        sudo -H pip install python-dateutil
        sudo -H pip install devlib
        sudo -H pip install pandas
        sudo -H pip install louie
        sudo -H pip install wrapt
        sudo -H pip install requests
        sudo -H pip install colorama
        sudo -H pip install future

Some of these may also be available in your distro's repositories, e.g. ::

        sudo apt-get install python-serial

Distro package versions tend to be older, so pip installation is recommended.
However, pip will always download and try to build the source, so in some
situations distro binaries may provide an easier fall back. Please also note that
distro package names may differ from pip packages.


Optional Python Packages
------------------------

.. note:: Unlike the mandatory dependencies in the previous section,
          pip will *not* install these automatically, so you will have
          to explicitly install them if/when you need them.

In addition to the mandatory packages listed in the previous sections, some WA
functionality (e.g. certain plugins) may have additional dependencies. Since
they are not necessary to be able to use most of WA, they are not made mandatory
to simplify initial WA installation. If you try to use an plugin that has
additional, unmet dependencies, WA will tell you before starting the run, and
you can install it then. They are listed here for those that would rather
install them upfront (e.g. if you're planning to use WA to an environment that
may not always have Internet access).

  * nose
  * mock
  * daqpower
  * sphinx
  * sphinx_rtd_theme
  * psycopg2-binary



.. _github:

Installing
==========

Installing the latest released version from PyPI (Python Package Index)::

       sudo -H pip install wa

This will install WA along with its mandatory dependencies. If you would like to
install all optional dependencies at the same time, do the following instead::

       sudo -H pip install wa[all]


Alternatively, you can also install the latest development version from GitHub
(you will need git installed for this to work)::

       git clone git@github.com:ARM-software/workload-automation.git workload-automation
       cd workload-automation
       sudo -H python setup.py install

.. note:: Please note that if using pip to install from github this will most
          likely result in an older and incompatible version of devlib being
          installed alongside WA. If you wish to use pip please also manually
          install the latest version of
          `devlib <https://github.com/ARM-software/devlib>`_.

.. note:: Please note that while a `requirements.txt` is included, this is
          designed to be a reference of known working packages rather to than to
          be used as part of a standard installation. The version restrictions
          in place as part of `setup.py` should automatically ensure the correct
          packages are install however if encountering issues please try
          updating/downgrading to the package versions list within.


If the above succeeds, try ::

        wa --version

Hopefully, this should output something along the lines of ::

        "Workload Automation version $version".

.. _dockerfile:

Dockerfile
============

As an alternative we also provide a Dockerfile that will create an image called
wadocker, and is preconfigured to run WA and devlib. Please note that the build
process automatically accepts the licenses for the Android SDK, so please be
sure that you are willing to accept these prior to building and running the
image in a container.

The Dockerfile can be found in the "extras" directory or online at
`<https://github.com/ARM-software /workload- automation/blob/next/extras/Dockerfile>`_
which contains additional information about how to build and to use the file.


(Optional) Post Installation
============================

Some WA plugins have additional dependencies that need to be
satisfied before they can be used. Not all of these can be provided with WA and
so will need to be supplied by the user. They should be placed into
``~/.workload_automation/dependencies/<extension name>`` so that WA can find
them (you may need to create the directory if it doesn't already exist). You
only need to provide the dependencies for workloads you want to use.

.. _apk_files:

APK Files
---------

APKs are application packages used by Android. These are necessary to install on
a device when running an :ref:`ApkWorkload <apk-workload>` or derivative. Please
see the workload description using the :ref:`show <show-command>` command to see
which version of the apk the UI automation has been tested with and place the
apk in the corresponding workloads dependency directory. Automation may also work
with other versions (especially if it's only a minor or revision difference --
major version differences are more likely to contain incompatible UI changes)
but this has not been tested. As a general rule we do not guarantee support for
the latest version of an app and they are updated on an as needed basis. We do
however attempt to support backwards compatibility with previous major releases
however beyond this support will likely be dropped.


Gaming Workloads
----------------

Some workloads (games, demos, etc) cannot be automated using Android's
UIAutomator framework because they render the entire UI inside a single OpenGL
surface. For these, an interaction session needs to be recorded so that it can
be played back by WA. These recordings are device-specific, so they would need
to be done for each device you're planning to use. The tool for doing is
``revent`` and it is packaged with WA. You can find instructions on how to use
it in the :ref:`How To <revent_files_creation>` section.

This is the list of workloads that rely on such recordings:

+------------------+
| angrybirds_rio   |
+------------------+
| templerun2       |
+------------------+


+------------------+

.. _assets_repository:

Maintaining Centralized Assets Repository
-----------------------------------------

If there are multiple users within an organization that may need to deploy
assets for WA plugins, that organization may wish to maintain a centralized
repository of assets that individual WA installs will be able to automatically
retrieve asset files from as they are needed. This repository can be any
directory on a network filer that mirrors the structure of
``~/.workload_automation/dependencies``, i.e. has a subdirectories named after
the plugins which assets they contain. Individual WA installs can then set
``remote_assets_path`` setting in their config to point to the local mount of
that location.


(Optional) Uninstalling
=======================

If you have installed Workload Automation via ``pip`` and wish to remove it, run this command to
uninstall it::

    sudo -H pip uninstall wa

.. Note:: This will *not* remove any user configuration (e.g. the ~/.workload_automation directory)


(Optional) Upgrading
====================

To upgrade Workload Automation to the latest version via ``pip``, run::

    sudo -H pip install --upgrade --no-deps wa
