.. _platform:

Platform
========

:class:`Platform`\ s describe the system underlying the OS. They encapsulate
hardware- and firmware-specific details. In most cases, the generic
:class:`Platform` class, which gets used if a platform is not explicitly
specified on :class:`Target` creation, will be sufficient. It will automatically
query as much platform information (such CPU topology, hardware model, etc) if
it was not specified explicitly by the user.


.. class:: Platform(name=None, core_names=None, core_clusters=None,\
                    big_core=None, model=None, modules=None)

    :param name: A user-friendly identifier for the platform.
    :param core_names: A list of CPU core names in the order they appear
                       registered with the OS. If they are not specified,
                       they will be queried at run time.
    :param core_clusters: Alist with cluster ids of each core (starting with
                          0). If this is not specified, clusters will be
                          inferred from core names (cores with the same name are
                          assumed to be in a cluster).
    :param big_core: The name of the big core in a big.LITTLE system. If this is
                     not specified it will be inferred (on systems with exactly 
                     two clasters).
    :param model: Model name of the hardware system. If this is not specified it
                  will be queried at run time.
    :param modules: Modules with additional functionality supported by the
                    platfrom (e.g. for handling flashing, rebooting, etc). These
                    would be added to the Target's modules. (See :ref:`modules`\ ).


Versatile Express
-----------------

The generic platform may be extended to support hardware- or
infrastructure-specific functionality. Platforms exist for ARM
VersatileExpress-based :class:`Juno` and :class:`TC2` development boards. In
addition to the standard :class:`Platform` parameters above, these platfroms
support additional configuration:


.. class:: VersatileExpressPlatform

    Normally, this would be instatiated via one of its derived classes
    (:class:`Juno` or :class:`TC2`) that set appropriate defaults for some of
    the parameters.

    :param serial_port: Identifies the serial port (usual a /dev node) on which the
                        device is connected.
    :param baudrate: Baud rate for the serial connection. This defaults to
                     ``115200`` for :class:`Juno` and ``38400`` for
                     :class:`TC2`.
    :param vemsd_mount: Mount point for the VEMSD (Versatile Express MicroSD card
                        that is used for board configuration files and firmware
                        images). This defaults to ``"/media/JUNO"`` for
                        :class:`Juno` and ``"/media/VEMSD"`` for :class:`TC2`,
                        though you would most likely need to change this for
                        your setup as it would depend both on the file system
                        label on the MicroSD card, and on how the card was
                        mounted on the host system.
    :param hard_reset_method: Specifies the method for hard-resetting the devices
                            (e.g. if it becomes unresponsive and normal reboot
                            method doesn not work). Currently supported methods
                            are:

                            :dtr: reboot by toggling DTR line on the serial
                                  connection (this is enabled via a DIP switch
                                  on the board).
                            :reboottxt: reboot by writing a filed called
                                        ``reboot.txt`` to the root of the VEMSD
                                        mount (this is enabled via board
                                        configuration file).

                            This defaults to ``dtr`` for :class:`Juno` and
                            ``reboottxt`` for :class:`TC2`.
    :param bootloader: Specifies the bootloader configuration used by the board.
                      The following values are currently supported:

                       :uefi: Boot via UEFI menu, by selecting the entry
                              specified by ``uefi_entry`` paramter. If this
                              entry does not exist, it will be automatically
                              created based on values provided for ``image``,
                              ``initrd``, ``fdt``, and ``bootargs`` parameters.
                       :uefi-shell: Boot by going via the UEFI shell.
                       :u-boot: Boot using Das U-Boot.
                       :bootmon: Boot directly via Versatile Express Bootmon
                                 using the values provided for ``image``, 
                                 ``initrd``, ``fdt``, and ``bootargs`` 
                                 parameters.

                      This defaults to ``u-boot`` for :class:`Juno` and
                      ``bootmon`` for :class:`TC2`.
    :param flash_method: Specifies how the device is flashed. Currently, only
                        ``"vemsd"`` method is supported, which flashes by
                        writing firmware images to an appropriate location on
                        the VEMSD.
    :param image: Specfies the kernel image name for ``uefi``  or ``bootmon`` boot.
    :param fdt: Specifies the device tree blob for  ``uefi``  or ``bootmon`` boot.
    :param initrd: Specifies the ramdisk image for  ``uefi`` or ``bootmon`` boot.
    :param bootargs: Specifies the boot arguments that will be pass to the
                     kernel by the bootloader.
    :param uefi_entry: Then name of the UEFI entry to be used/created by
                       ``uefi`` bootloader.
    :param ready_timeout: Timeout, in seconds, for the time it takes the
                          platform to become ready to accept connections. Note:
                          this does not mean that the system is fully booted;
                          just that the services needed to establish a
                          connection (e.g. sshd or adbd) are up.

