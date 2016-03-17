============
Installation
============

.. module:: wlauto

This page describes how to install Workload Automation 2.


Prerequisites
=============

Operating System
----------------

WA runs on a native Linux install. It was tested with Ubuntu 12.04,
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
        Android Debug Bridge version 1.0.31

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

Workload Automation 2 requires Python 2.7 (Python 3 is not supported at the moment).


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

                  sudo chmod -R a+r /usr/local/lib/python2.7/dist-packagessudo 
                  find /usr/local/lib/python2.7/dist-packages -type d -exec chmod a+x {} \;

          (The paths above will work for Ubuntu; they may need to be adjusted
          for other distros).

Python Packages
---------------

.. note:: pip should automatically download and install missing dependencies,
          so if you're using pip, you can skip this section.

Workload Automation 2 depends on the following additional libraries:

  * pexpect
  * docutils
  * pySerial
  * pyYAML
  * python-dateutil

You can install these with pip::

        sudo -H pip install pexpect
        sudo -H pip install pyserial
        sudo -H pip install pyyaml
        sudo -H pip install docutils
        sudo -H pip install python-dateutil

Some of these may also be available in your distro's repositories, e.g. ::

        sudo apt-get install python-serial

Distro package versions tend to be older, so pip installation is recommended.
However, pip will always download and try to build the source, so in some
situations distro binaries may provide an easier fall back. Please also note that
distro package names may differ from pip packages.


Optional Python Packages
------------------------

.. note:: unlike the mandatory dependencies in the previous section,
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
  * pandas
  * PyDAQmx
  * pymongo
  * jinja2


.. note:: Some packages have C plugins and will require Python development
          headers to install. You can get those by installing ``python-dev``
          package in apt on Ubuntu (or the equivalent for your distribution).


Installing
==========

Installing the latest released version from PyPI (Python Package Index)::

       sudo -H pip install wlauto

This will install WA along with its mandatory dependencies. If you would like to
install all optional dependencies at the same time, do the following instead::

       sudo -H pip install wlauto[all]

Alternatively, you can also install the latest development version from GitHub
(you will need git installed for this to work)::

       git clone git@github.com:ARM-software/workload-automation.git workload-automation
       sudo -H pip install ./workload-automation



If the above succeeds, try ::

        wa --version

Hopefully, this should output something along the lines of "Workload Automation
version $version".


(Optional) Post Installation
============================

Some WA plugins have additional dependencies that need to be
statisfied before they can be used. Not all of these can be provided with WA and
so will need to be supplied by the user. They should be placed into
``~/.workload_uatomation/dependencies/<extenion name>`` so that WA can find
them (you may need to create the directory if it doesn't already exist). You
only need to provide the dependencies for workloads you want to use.


APK Files
---------

APKs are applicaton packages used by Android. These are necessary to install an
application onto devices that do not have Google Play (e.g. devboards running
AOSP). The following is a list of workloads that will need one, including the
version(s) for which UI automation has been tested. Automation may also work
with other versions (especially if it's only a minor or revision difference --
major version differens are more likely to contain incompatible UI changes) but
this has not been tested.

================ ============================================ ========================= ============ ============
workload         package                                      name                      version code version name
================ ============================================ ========================= ============ ============
andebench        com.eembc.coremark                           AndEBench                       v1383a         1383
angrybirds       com.rovio.angrybirds                         Angry Birds                      2.1.1         2110
angrybirds_rio   com.rovio.angrybirdsrio                      Angry Birds                      1.3.2         1320
anomaly2         com.elevenbitstudios.anomaly2Benchmark       A2 Benchmark                       1.1           50
antutu           com.antutu.ABenchMark                        AnTuTu Benchmark                   5.3      5030000
antutu           com.antutu.ABenchMark                        AnTuTu Benchmark                 3.3.2         3322
antutu           com.antutu.ABenchMark                        AnTuTu Benchmark                 4.0.3      4000300
benchmarkpi      gr.androiddev.BenchmarkPi                    BenchmarkPi                       1.11            5
caffeinemark     com.flexycore.caffeinemark                   CaffeineMark                     1.2.4            9
castlebuilder    com.ettinentertainment.castlebuilder         Castle Builder                     1.0            1
castlemaster     com.alphacloud.castlemaster                  Castle Master                     1.09          109
cfbench          eu.chainfire.cfbench                         CF-Bench                           1.2            7
citadel          com.epicgames.EpicCitadel                    Epic Citadel                      1.07       901107
dungeondefenders com.trendy.ddapp                             Dungeon Defenders                 5.34           34
facebook         com.facebook.katana                          Facebook                           3.4       258880
geekbench        ca.primatelabs.geekbench2                    Geekbench 2                      2.2.7       202007
geekbench        com.primatelabs.geekbench3                   Geekbench 3                      3.0.0          135
glb_corporate    net.kishonti.gfxbench                        GFXBench                         3.0.0            1
glbenchmark      com.glbenchmark.glbenchmark25                GLBenchmark 2.5                    2.5            4
glbenchmark      com.glbenchmark.glbenchmark27                GLBenchmark 2.7                    2.7            1
gunbros2         com.glu.gunbros2                             GunBros2                         1.2.2          122
ironman          com.gameloft.android.ANMP.GloftIMHM          Iron Man 3                       1.3.1         1310
krazykart        com.polarbit.sg2.krazyracers                 Krazy Kart Racing                1.2.7          127
linpack          com.greenecomputing.linpackpro               Linpack Pro for Android          1.2.9           31
nenamark         se.nena.nenamark2                            NenaMark2                          2.4            5
peacekeeper      com.android.chrome                           Chrome                    18.0.1025469      1025469
peacekeeper      org.mozilla.firefox                          Firefox                           23.0   2013073011
quadrant         com.aurorasoftworks.quadrant.ui.professional Quadrant Professional              2.0      2000000
realracing3      com.ea.games.r3_row                          Real Racing 3                    1.3.5         1305
smartbench       com.smartbench.twelve                        Smartbench 2012                  1.0.0            5
sqlite           com.redlicense.benchmark.sqlite              RL Benchmark                       1.3            5
templerun        com.imangi.templerun                         Temple Run                       1.0.8           11
thechase         com.unity3d.TheChase                         The Chase                          1.0            1
truckerparking3d com.tapinator.truck.parking.bus3d            Truck Parking 3D                   2.5            7
vellamo          com.quicinc.vellamo                          Vellamo                            3.0         3001
vellamo          com.quicinc.vellamo                          Vellamo                          2.0.3         2003
videostreaming   tw.com.freedi.youtube.player                 FREEdi YT Player                2.1.13           79
================ ============================================ ========================= ============ ============

Gaming Workloads
----------------

Some workloads (games, demos, etc) cannot be automated using Android's
UIAutomator framework because they render the entire UI inside a single OpenGL
surface. For these, an interaction session needs to be recorded so that it can
be played back by WA. These recordings are device-specific, so they would need
to be done for each device you're planning to use. The tool for doing is
``revent`` and it is packaged with WA. You can find instructions on how to use
it :ref:`here <revent_files_creation>`.

This is the list of workloads that rely on such recordings:

+------------------+
| angrybirds       |
+------------------+
| angrybirds_rio   |
+------------------+
| anomaly2         |
+------------------+
| castlebuilder    |
+------------------+
| castlemastera    |
+------------------+
| citadel          |
+------------------+
| dungeondefenders |
+------------------+
| gunbros2         |
+------------------+
| ironman          |
+------------------+
| krazykart        |
+------------------+
| realracing3      |
+------------------+
| templerun        |
+------------------+
| truckerparking3d |
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
    
    sudo -H pip uninstall wlauto

.. Note:: This will *not* remove any user configuration (e.g. the ~/.workload_automation directory)


(Optional) Upgrading
====================

To upgrade Workload Automation to the latest version via ``pip``, run::
    
    sudo -H pip install --upgrade --no-deps wlauto
