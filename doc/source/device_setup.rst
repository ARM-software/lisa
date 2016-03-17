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
+++++++

General Device Setup
--------------------

You can specify the device interface by setting ``device`` setting in
``~/.workload_automation/config.py``. Available interfaces can be viewed by
running ``wa list devices`` command. If you don't see your specific device
listed (which is likely unless you're using one of the ARM-supplied platforms), then
you should use ``generic_android`` interface (this is set in the config by
default).

.. code-block:: python

        device = 'generic_android'

The device interface may be configured through ``device_config`` setting, who's
value is a ``dict`` mapping setting names to their values. You can find the full
list of available parameter by looking up your device interface in the
:ref:`devices` section of the documentation. Some of the most common parameters
you might want to change are outlined below.

.. confval:: adb_name

   If you have multiple Android devices connected to the host machine, you will
   need to set this to indicate to WA which device you want it to use.

.. confval:: working_directory

   WA needs a "working" directory on the device which it will use for collecting
   traces, caching assets it pushes to the device, etc. By default, it will
   create one under ``/sdcard`` which should be mapped and writable on standard
   Android builds. If this is not the case for your device, you will need to
   specify an alternative working directory (e.g. under ``/data/local``).

.. confval:: scheduler

   This specifies the scheduling mechanism (from the perspective of core layout)
   utilized by the device). For recent big.LITTLE devices, this should generally
   be "hmp" (ARM Hetrogeneous Mutli-Processing); some legacy development
   platforms might have Linaro IKS kernels, in which case it should be "iks".
   For homogeneous (single-cluster) devices, it should be "smp". Please see
   ``scheduler`` parameter  in the ``generic_android`` device documentation for
   more details.

.. confval:: core_names

   This and ``core_clusters`` need to be set if you want to utilize some more
   advanced WA functionality (like setting of core-related runtime parameters
   such as governors, frequencies, etc). ``core_names`` should be a list of
   core names matching the order in which they are exposed in sysfs. For
   example, ARM TC2 SoC is a 2x3 big.LITTLE system; its core_names would be
   ``['a7', 'a7', 'a7', 'a15', 'a15']``, indicating that cpu0-cpu2 in cpufreq
   sysfs structure are A7's and cpu3 and cpu4 are A15's.

.. confval:: core_clusters

   If ``core_names`` is defined, this must also be defined. This is a list of
   integer values indicating the cluster the corresponding core in
   ``cores_names`` belongs to. For example, for TC2, this would be
   ``[0, 0, 0, 1, 1]``, indicating that A7's are on cluster 0 and A15's are on
   cluster 1.

A typical ``device_config`` inside ``config.py`` may look something like


.. code-block:: python

        device_config = dict(
                'adb_name'='0123456789ABCDEF',
                'working_direcory'='/sdcard/wa-working',
                'core_names'=['a7', 'a7', 'a7', 'a15', 'a15'],
                'core_clusters'=[0, 0, 0, 1, 1],
                # ...
        )

.. _configuring-android:

Configuring Android
-------------------

There are a few additional tasks you may need to perform once you have a device
booted into Android (especially if this is an initial boot of a fresh OS
deployment):

        - You have gone through FTU (first time usage) on the home screen and
          in the apps menu.
        - You have disabled the screen lock.
        - You have set sleep timeout to the highest possible value (30 mins on
          most devices).
        - You have disabled brightness auto-adjust and have set the brightness
          to a fixed level.
        - You have set the locale language to "English" (this is important for
          some workloads in which UI automation looks for specific text in UI
          elements).

TC2 Setup
---------

This section outlines how to setup ARM TC2 development platform to work with WA.

Pre-requisites
~~~~~~~~~~~~~~

You can obtain the full set of images for TC2 from Linaro:

https://releases.linaro.org/latest/android/vexpress-lsk. 

For the easiest setup, follow the instructions on the "Firmware" and "Binary
Image Installation" tabs on that page.

.. note:: The default ``reboot_policy`` in ``config.py`` is to not reboot. With
          this WA will assume that the device is already booted into Android
          prior to WA being invoked. If you want to WA to do the initial boot of
          the TC2, you will have to change reboot policy to at least
          ``initial``.


Setting Up Images
~~~~~~~~~~~~~~~~~

.. note:: Make sure that both DIP switches near the black reset button on TC2
          are up (this is counter to the Linaro guide that instructs to lower
          one of the switches).

.. note:: The TC2 must have an Ethernet connection.


If you have followed the setup instructions on the Linaro page, you should have
a USB stick or an SD card with the file system, and internal microSD on the
board (VEMSD) with the firmware images. The default Linaro configuration is to
boot from the image on the boot partition in the file system you have just
created. This is not supported by WA, which expects the image to be in NOR flash
on the board. This requires you to copy the images from the boot partition onto
the internal microSD card.

Assuming the boot partition of the Linaro file system is mounted on
``/media/boot`` and the internal microSD  is mounted on ``/media/VEMSD``, copy
the following images::

        cp /media/boot/zImage /media/VEMSD/SOFTWARE/kern_mp.bin
        cp /media/boot/initrd /media/VEMSD/SOFTWARE/init_mp.bin
        cp /media/boot/v2p-ca15-tc2.dtb /media/VEMSD/SOFTWARE/mp_a7bc.dtb

Optionally
##########

The default device tree configuration the TC2 is to boot on the A7 cluster. It
is also possible to configure the device tree to boot on the A15 cluster, or to
boot with one of the clusters disabled (turning TC2 into an A7-only or A15-only
device). Please refer to the "Firmware" tab on the Linaro paged linked above for
instructions on how to compile the appropriate device tree configurations.

WA allows selecting between these configurations using ``os_mode`` boot
parameter of the TC2 device interface. In order for this to work correctly,
device tree files for the A15-bootcluster, A7-only and A15-only configurations
should be copied into ``/media/VEMSD/SOFTWARE/`` as ``mp_a15bc.dtb``,
``mp_a7.dtb`` and ``mp_a15.dtb`` respectively.

This is entirely optional. If you're not planning on switching boot cluster
configuration, those files do not need to be present in VEMSD.

config.txt
##########

Also, make sure that ``USB_REMOTE`` setting in ``/media/VEMSD/config.txt`` is set
to ``TRUE`` (this will allow rebooting the device by writing reboot.txt to
VEMSD). ::

    USB_REMOTE: TRUE                 ;Selects remote command via USB
    

TC2-specific device_config settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a few settings that may need to be set in ``device_config`` inside
your ``config.py`` which are specific to TC2:

.. note:: TC2 *does not* accept most "standard" android ``device_config``
          settings.
          
adb_name
        If you're running WA with reboots disabled (which is the default reboot
        policy), you will need to manually run ``adb connect`` with TC2's IP
        address and set this.

root_mount
        WA expects TC2's internal microSD to be mounted on the host under
        ``/media/VEMSD``. If this location is different, it needs to be specified
        using this setting.

boot_firmware
        WA defaults to try booting using UEFI, which will require some additional
        firmware from ARM that may not be provided with Linaro releases (see the
        UEFI and PSCI section below). If you do not have those images, you will
        need to set ``boot_firmware`` to ``bootmon``.

fs_medium
        TC2's file system can reside either on an SD card or on a USB stick. Boot
        configuration is different depending on this. By default,  WA expects it
        to be on ``usb``; if you are using and SD card, you should set this to
        ``sd``.

bm_image
        Bootmon image that comes as part of TC2 firmware periodically gets
        updated. At the time of the release, ``bm_v519r.axf`` was used by
        ARM. If you are using a more recent image, you will need to set this
        indicating the image name (just the name of the actual file, *not* the
        path). Note: this setting only applies if using ``bootmon`` boot
        firmware.

serial_device
        WA will assume TC2 is connected on ``/dev/ttyS0`` by default. If the
        serial port is different, you will need to set this.


UEFI and PSCI
~~~~~~~~~~~~~

UEFI is a boot firmware alternative to bootmon. Currently UEFI is coupled with PSCI (Power State Coordination Interface). That means
that in order to use PSCI, UEFI has to be the boot firmware. Currently the reverse dependency is true as well (for TC2). Therefore
using UEFI requires enabling PSCI.

In case you intend to use uefi/psci mode instead of bootmon, you will need two additional files: tc2_sec.bin and tc2_uefi.bin.
after obtaining those files, place them inside /media/VEMSD/SOFTWARE/ directory as such::

    cp tc2_sec.bin /media/VEMSD/SOFTWARE/
    cp tc2_uefi.bin /media/VEMSD/SOFTWARE/


Juno Setup
----------

.. note:: At the time of writing, the Android software stack on Juno was still
          very immature. Some workloads may not run, and there maybe stability
          issues with the device.


The full software stack can be obtained from Linaro:

https://releases.linaro.org/14.08/members/arm/android/images/armv8-android-juno-lsk

Please follow the instructions on the "Binary Image Installation" tab on that
page. More up-to-date firmware and kernel may also be obtained by registered
members from ARM Connected Community: http://www.arm.com/community/ (though this
is not guaranteed to work with the Linaro file system).

UEFI
~~~~

Juno uses UEFI_ to boot the kernel image.  UEFI supports multiple boot
configurations, and presents a menu on boot to select (in default configuration
it will automatically boot the first entry in the menu if not interrupted before
a timeout). WA will look for a specific entry in the UEFI menu
(``'WA'`` by default, but that may be changed by setting ``uefi_entry`` in the
``device_config``). When following the UEFI instructions on the above Linaro
page, please make sure to name the entry appropriately (or to correctly set the
``uefi_entry``).

.. _UEFI: http://en.wikipedia.org/wiki/UEFI

There are two supported way for Juno to discover kernel images through UEFI. It
can either load them from NOR flash on the board, or form boot partition on the
file system. The setup described on the Linaro page uses the boot partition
method.

If WA does not find the UEFI entry it expects, it will create one. However, it
will assume that the kernel image resides in NOR flash, which means it will not
work with Linaro file system. So if you're replicating the Linaro setup exactly,
you will need to create the entry manually, as outline on the above-linked page.

Rebooting
~~~~~~~~~

At the time of writing, normal Android reboot did not work properly on Juno
Android, causing the device to crash into an irrecoverable state. Therefore, WA
will perform a hard reset to reboot the device. It will attempt to do this by
toggling the DTR line on the serial connection to the device. In order for this
to work, you need to make sure that SW1 configuration switch on the back panel of
the board (the right-most DIP switch) is toggled *down*.


Linux
+++++

General Device Setup
--------------------

You can specify the device interface by setting ``device`` setting in
``~/.workload_automation/config.py``. Available interfaces can be viewed by
running ``wa list devices`` command. If you don't see your specific device
listed (which is likely unless you're using one of the ARM-supplied platforms), then
you should use ``generic_linux`` interface (this is set in the config by
default).

.. code-block:: python

        device = 'generic_linux'

The device interface may be configured through ``device_config`` setting, who's
value is a ``dict`` mapping setting names to their values. You can find the full
list of available parameter by looking up your device interface in the
:ref:`devices` section of the documentation. Some of the most common parameters
you might want to change are outlined below.

Currently, the only only supported method for talking to a Linux device is over
SSH. Device configuration must specify the parameters need to establish the
connection.

.. confval:: host

   This should be either the the DNS name or IP address of the device.

.. confval:: username

   The login name of the user on the device that WA will use. This user should 
   have a home directory (unless an alternative working directory is specified
   using ``working_directory`` config -- see below), and, for full
   functionality, the user should have sudo rights (WA will be able to use
   sudo-less acounts but some instruments or workload may not work).

.. confval:: password

   Password for the account on the device. Either this of a ``keyfile`` (see
   below) must be specified.

.. confval:: keyfile

   If key-based authentication is used, this may be used to specify the SSH identity 
   file instead of the password.

.. confval:: property_files

   This is a list of paths that will be pulled for each WA run into the __meta
   subdirectory in the results. The intention is to collect meta-data about the 
   device that may aid in reporducing the results later. The paths specified do
   not have to exist on the device (they will be ignored if they do not). The
   default list is ``['/proc/version', '/etc/debian_version', '/etc/lsb-release', '/etc/arch-release']``


In addition, ``working_directory``, ``scheduler``, ``core_names``, and
``core_clusters`` can also be specified and have the same meaning as for Android
devices (see above).

A typical ``device_config`` inside ``config.py`` may look something like


.. code-block:: python

        device_config = dict(
                host='192.168.0.7',
                username='guest',
                password='guest',
                core_names=['a7', 'a7', 'a7', 'a15', 'a15'],
                core_clusters=[0, 0, 0, 1, 1],
                # ...
        )


Related Settings
++++++++++++++++

Reboot Policy
-------------

This indicates when during WA execution the device will be rebooted. By default
this is set to ``never``, indicating that WA will not reboot the device. Please
see ``reboot_policy`` documentation in :ref:`configuration-specification` for

more details.

Execution Order
---------------

``execution_order`` defines the order in which WA will execute workloads.
``by_iteration`` (set by default) will execute the first iteration of each spec
first, followed by the second iteration of each spec (that defines more than one
iteration) and so forth. The alternative  will loop through all iterations for
the first first spec first, then move on to second spec, etc. Again, please see
:ref:`configuration-specification` for more details.


Adding a new device interface
+++++++++++++++++++++++++++++

If you are working with a particularly unusual device (e.g. a early stage
development board) or need to be able to handle some quirk of your Android build,
configuration available in ``generic_android`` interface may not be enough for
you. In that case, you may need to write a custom interface for your device. A
device interface is an ``Plugin`` (a plug-in) type in WA and is implemented
similar to other plugins (such as workloads or instruments). Pleaser refer to
:ref:`adding_a_device` section for information on how this may be done.
