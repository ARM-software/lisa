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
    :param core_clusters: A list with cluster ids of each core (starting with
                          0). If this is not specified, clusters will be
                          inferred from core names (cores with the same name are
                          assumed to be in a cluster).
    :param big_core: The name of the big core in a big.LITTLE system. If this is
                     not specified it will be inferred (on systems with exactly
                     two clusters).
    :param model: Model name of the hardware system. If this is not specified it
                  will be queried at run time.
    :param modules: Modules with additional functionality supported by the
                    platform (e.g. for handling flashing, rebooting, etc). These
                    would be added to the Target's modules. (See :ref:`modules`\ ).


Versatile Express
-----------------

The generic platform may be extended to support hardware- or
infrastructure-specific functionality. Platforms exist for ARM
VersatileExpress-based :class:`Juno` and :class:`TC2` development boards. In
addition to the standard :class:`Platform` parameters above, these platforms
support additional configuration:


.. class:: VersatileExpressPlatform

    Normally, this would be instantiated via one of its derived classes
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
                            method doesn't not work). Currently supported methods
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
                              specified by ``uefi_entry`` parameter. If this
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


.. _gem5-platform:

Gem5 Simulation Platform
------------------------

By initialising a Gem5SimulationPlatform, devlib will start a gem5 simulation (based upon the
arguments the user provided) and then connect to it using :class:`Gem5Connection`.
Using the methods discussed above, some methods of the :class:`Target` will be altered
slightly to better suit gem5.

.. class:: Gem5SimulationPlatform(name, host_output_dir, gem5_bin, gem5_args, gem5_virtio, gem5_telnet_port=None)

    During initialisation the gem5 simulation will be kicked off (based upon the arguments
    provided by the user) and the telnet port used by the gem5 simulation will be intercepted
    and stored for use by the :class:`Gem5Connection`.

    :param name: Platform name

    :param host_output_dir: Path on the host where the gem5 outputs will be placed (e.g. stats file)

    :param gem5_bin: gem5 binary

    :param gem5_args: Arguments to be passed onto gem5 such as config file etc.

    :param gem5_virtio: Arguments to be passed onto gem5 in terms of the virtIO device used
                        to transfer files between the host and the gem5 simulated system.

    :param gem5_telnet_port: Not yet in use as it would be used in future implementations
                             of devlib in which the user could use the platform to pick
                             up an existing and running simulation.


.. method:: Gem5SimulationPlatform.init_target_connection([target])

    Based upon the OS defined in the :class:`Target`, the type of :class:`Gem5Connection`
    will be set (:class:`AndroidGem5Connection` or :class:`AndroidGem5Connection`).

.. method:: Gem5SimulationPlatform.update_from_target([target])

    This method provides specific setup procedures for a gem5 simulation. First of all, the m5
    binary will be installed on the guest (if it is not present). Secondly, three methods
    in the :class:`Target` will be monkey-patched:

            - **reboot**: this is not supported in gem5
            - **reset**: this is not supported in gem5
            - **capture_screen**: gem5 might already have screencaps so the
              monkey-patched method will first try to
              transfer the existing screencaps.
              In case that does not work, it will fall back
              to the original :class:`Target` implementation
              of :func:`capture_screen`.

    Finally, it will call the parent implementation of :func:`update_from_target`.

.. method:: Gem5SimulationPlatform.setup([target])

    The m5 binary be installed, if not yet installed on the gem5 simulated system.
    It will also resize the gem5 shell, to avoid line wrapping issues.
