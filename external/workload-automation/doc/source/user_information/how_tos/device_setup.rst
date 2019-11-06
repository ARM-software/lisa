.. _setting-up-a-device:

Setting Up A Device
===================

WA should work with most Android devices out-of-the box, as long as the device
is discoverable by ``adb`` (i.e. gets listed when you run ``adb devices``). For
USB-attached devices, that should be the case; for network devices, ``adb connect``
would need to be invoked with the IP address of the device. If there is only one
device connected to the host running WA, then no further configuration should be
necessary (though you may want to :ref:`tweak some Android settings <configuring-android>`\ ).

If you have multiple devices connected, have a non-standard Android build (e.g.
on a development board), or want to use of the more advanced WA functionality,
further configuration will be required.

Android
-------

.. _android-general-device-setup:

General Device Setup
^^^^^^^^^^^^^^^^^^^^

You can specify the device interface by setting ``device`` setting in a
``config`` file or section. Available interfaces can be viewed by running ``wa
list targets`` command. If you don't see your specific platform listed (which is
likely unless you're using one of the Arm-supplied platforms), then you should
use ``generic_android`` interface (this is what is used by the default config).

.. code-block:: yaml

        device: generic_android

The device interface may be configured through ``device_config`` setting, who's
value is a ``dict`` mapping setting names to their values. Some of the most
common parameters you might want to change are outlined below.

:device: If you have multiple Android devices connected to the host machine, you will
   need to set this to indicate to WA which device you want it to use. The will
   be the adb name the is displayed when running ``adb devices``

:working_directory: WA needs a "working" directory on the device which it will use for collecting
   traces, caching assets it pushes to the device, etc. By default, it will
   create one under ``/sdcard`` which should be mapped and writable on standard
   Android builds. If this is not the case for your device, you will need to
   specify an alternative working directory (e.g. under ``/data/local``).

:load_default_modules: A number of "default" modules (e.g. for cpufreq
  subsystem) are loaded automatically, unless explicitly disabled. If you
  encounter an issue with one of the modules then this setting can be set to
  ``False`` and any specific modules that you require can be request via the
  ``modules`` entry.

:modules: A list of additional modules to be installed for the target. Devlib
  implements functionality for particular subsystems as modules. If additional
  modules need to be loaded, they may be specified using this parameter.

  Please see the `devlib documentation <http://devlib.readthedocs.io/en/latest/modules.html>`_
  for information on the available modules.

.. _core-names:

:core_names: ``core_names`` should be a list of core names matching the order in which
   they are exposed in sysfs. For example, Arm TC2 SoC is a 2x3 big.LITTLE
   system; its core_names would be ``['a7', 'a7', 'a7', 'a15', 'a15']``,
   indicating that cpu0-cpu2 in cpufreq sysfs structure are A7's and cpu3 and
   cpu4 are A15's.

   .. note:: This should not usually need to be provided as it will be
             automatically extracted from the target.


A typical ``device_config`` inside ``config.yaml`` may look something like


.. code-block:: yaml

        device_config:
                device: 0123456789ABCDEF
        # ...


or a more specific config could be:

.. code-block:: yaml

        device_config:
                device: 0123456789ABCDEF
                working_direcory: '/sdcard/wa-working'
                load_default_modules: True
                modules: ['hotplug', 'cpufreq']
                core_names : ['a7', 'a7', 'a7', 'a15', 'a15']
                # ...

.. _configuring-android:

Configuring Android
^^^^^^^^^^^^^^^^^^^

There are a few additional tasks you may need to perform once you have a device
booted into Android (especially if this is an initial boot of a fresh OS
deployment):

        - You have gone through FTU (first time usage) on the home screen and
          in the apps menu.
        - You have disabled the screen lock.
        - You have set sleep timeout to the highest possible value (30 mins on
          most devices).
        - You have set the locale language to "English" (this is important for
          some workloads in which UI automation looks for specific text in UI
          elements).


Juno Setup
----------

.. note:: At the time of writing, the Android software stack on Juno was still
          very immature. Some workloads may not run, and there maybe stability
          issues with the device.


The full software stack can be obtained from Linaro:

https://releases.linaro.org/android/images/lcr-reference-juno/latest/

Please follow the instructions on the "Binary Image Installation" tab on that
page. More up-to-date firmware and kernel may also be obtained by registered
members from ARM Connected Community: http://www.arm.com/community/ (though this
is not guaranteed to work with the Linaro file system).

UEFI
^^^^

Juno uses UEFI_ to boot the kernel image.  UEFI supports multiple boot
configurations, and presents a menu on boot to select (in default configuration
it will automatically boot the first entry in the menu if not interrupted before
a timeout). WA will look for a specific entry in the UEFI menu
(``'WA'`` by default, but that may be changed by setting ``uefi_entry`` in the
``device_config``). When following the UEFI instructions on the above Linaro
page, please make sure to name the entry appropriately (or to correctly set the
``uefi_entry``).

.. _UEFI: http://en.wikipedia.org/wiki/UEFI

There are two supported ways for Juno to discover kernel images through UEFI. It
can either load them from NOR flash on the board, or from the boot partition on
the file system. The setup described on the Linaro page uses the boot partition
method.

If WA does not find the UEFI entry it expects, it will create one. However, it
will assume that the kernel image resides in NOR flash, which means it will not
work with Linaro file system. So if you're replicating the Linaro setup exactly,
you will need to create the entry manually, as outline on the above-linked page.

Rebooting
^^^^^^^^^

At the time of writing, normal Android reboot did not work properly on Juno
Android, causing the device to crash into an irrecoverable state. Therefore, WA
will perform a hard reset to reboot the device. It will attempt to do this by
toggling the DTR line on the serial connection to the device. In order for this
to work, you need to make sure that SW1 configuration switch on the back panel of
the board (the right-most DIP switch) is toggled *down*.


Linux
-----

General Device Setup
^^^^^^^^^^^^^^^^^^^^

You can specify the device interface by setting ``device`` setting in a
``config`` file or section. Available interfaces can be viewed by running
``wa list targets`` command. If you don't see your specific platform listed
(which is likely unless you're using one of the Arm-supplied platforms), then
you should use ``generic_linux`` interface.

.. code-block:: yaml

        device: generic_linux

The device interface may be configured through ``device_config`` setting, who's
value is a ``dict`` mapping setting names to their values. Some of the most
common parameters you might want to change are outlined below.


:host: This should be either the the DNS name or IP address of the device.

:username: The login name of the user on the device that WA will use. This user should
   have a home directory (unless an alternative working directory is specified
   using ``working_directory`` config -- see below), and, for full
   functionality, the user should have sudo rights (WA will be able to use
   sudo-less acounts but some instruments or workload may not work).

:password: Password for the account on the device. Either this of a ``keyfile`` (see
   below) must be specified.

:keyfile: If key-based authentication is used, this may be used to specify the SSH identity
   file instead of the password.

:property_files: This is a list of paths that will be pulled for each WA run into the __meta
   subdirectory in the results. The intention is to collect meta-data about the
   device that may aid in reporducing the results later. The paths specified do
   not have to exist on the device (they will be ignored if they do not). The
   default list is ``['/proc/version', '/etc/debian_version', '/etc/lsb-release', '/etc/arch-release']``


In addition, ``working_directory``, ``core_names``, ``modules`` etc. can also
be specified and have the same meaning as for Android devices (see above).

A typical ``device_config`` inside ``config.yaml`` may look something like


.. code-block:: yaml

        device_config:
                host: 192.168.0.7
                username: guest
                password: guest
                # ...

Chrome OS
---------

General Device Setup
^^^^^^^^^^^^^^^^^^^^

You can specify the device interface by setting ``device`` setting in a
``config`` file or section. Available interfaces can be viewed by
running ``wa list targets`` command. If you don't see your specific platform
listed (which is likely unless you're using one of the Arm-supplied platforms), then
you should use ``generic_chromeos`` interface.

.. code-block:: yaml

        device: generic_chromeos

The device interface may be configured through ``device_config`` setting, who's
value is a ``dict`` mapping setting names to their values. The ChromeOS target
is essentially the same as a linux device and requires a similar setup, however
it also optionally supports connecting to an android container running on the
device which will be automatically detected if present. If the device supports
android applications then the android configuration is also supported. In order
to support this WA will open 2 connections to the device, one via SSH to
the main OS and another via ADB to the android container where a limited
subset of functionality can be performed.

In order to distinguish between the two connections some of the android specific
configuration has been renamed to reflect the destination.

:android_working_directory: WA needs a "working" directory on the device which it will use for collecting
   traces, caching assets it pushes to the device, etc. By default, it will
   create one under ``/sdcard`` which should be mapped and writable on standard
   Android builds. If this is not the case for your device, you will need to
   specify an alternative working directory (e.g. under ``/data/local``).


A typical ``device_config`` inside ``config.yaml`` for a ChromeOS device may
look something like

.. code-block:: yaml

        device_config:
                host: 192.168.0.7
                username: root
                android_working_direcory: '/sdcard/wa-working'
                # ...

.. note:: This assumes that your Chromebook is in developer mode and is
          configured to run an SSH server with the appropriate ssh keys added to the
          authorized_keys file on the device.


Related Settings
----------------

Reboot Policy
^^^^^^^^^^^^^

This indicates when during WA execution the device will be rebooted. By default
this is set to ``as_needed``, indicating that WA will only reboot the device if
it becomes unresponsive. Please see ``reboot_policy`` documentation in
:ref:`configuration-specification` for more details.

Execution Order
^^^^^^^^^^^^^^^

``execution_order`` defines the order in which WA will execute workloads.
``by_iteration`` (set by default) will execute the first iteration of each spec
first, followed by the second iteration of each spec (that defines more than one
iteration) and so forth. The alternative  will loop through all iterations for
the first first spec first, then move on to second spec, etc. Again, please see
:ref:`configuration-specification` for more details.


Adding a new target interface
-----------------------------

If you are working with a particularly unusual device (e.g. a early stage
development board) or need to be able to handle some quirk of your Android
build, configuration available in ``generic_android`` interface may not be
enough for you. In that case, you may need to write a custom interface for your
device. A device interface is an ``Extension`` (a plug-in) type in WA and is
implemented similar to other extensions (such as workloads or instruments).
Pleaser refer to the
:ref:`adding a custom target <adding-custom-target-example>` section for
information on how this may be done.
