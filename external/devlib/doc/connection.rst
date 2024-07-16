Connection
==========

A :class:`Connection` abstracts an actual physical connection to a device. The
first connection is created when :func:`Target.connect` method is called. If a
:class:`~devlib.target.Target` is used in a multi-threaded environment, it will
maintain a connection for each thread in which it is invoked. This allows
the same target object to be used in parallel in multiple threads.

:class:`Connection`\ s will be automatically created and managed by
:class:`~devlib.target.Target`\ s, so there is usually no reason to create one
manually. Instead, configuration for a :class:`Connection` is passed as
`connection_settings` parameter when creating a
:class:`~devlib.target.Target`. The connection to be used target is also
specified on instantiation by `conn_cls` parameter, though all concrete
:class:`~devlib.target.Target` implementations will set an appropriate
default, so there is typically no need to specify this explicitly.

:class:`Connection` classes are not a part of an inheritance hierarchy, i.e.
they do not derive from a common base. Instead, a :class:`Connection` is any
class that implements the following methods.


.. method:: push(self, sources, dest, timeout=None)

   Transfer a list of files from the host machine to the connected device.

   :param sources: list of paths on the host
   :param dest: path to the file or folder on the connected device.
   :param timeout: timeout (in seconds) for the transfer of each file; if the
       transfer does not complete within this period, an exception will be
       raised.

.. method:: pull(self, sources, dest, timeout=None)

   Transfer a list of files from the connected device to the host machine.

   :param sources: list of paths on the connected device.
   :param dest: path to the file or folder on the host
   :param timeout: timeout (in seconds) for the transfer for each file; if the
       transfer does not complete within this period, an exception will be
       raised.

.. method:: execute(self, command, timeout=None, check_exit_code=False, as_root=False, strip_colors=True, will_succeed=False)

   Execute the specified command on the connected device and return its output.

   :param command: The command to be executed.
   :param timeout: Timeout (in seconds) for the execution of the command. If
       specified, an exception will be raised if execution does not complete
       with the specified period.
   :param check_exit_code: If ``True`` the exit code (on connected device)
       from execution of the command will be checked, and an exception will be
       raised if it is not ``0``.
   :param as_root: The command will be executed as root. This will fail on
       unrooted connected devices.
   :param strip_colours: The command output will have colour encodings and
       most ANSI escape sequences striped out before returning.
   :param will_succeed: The command is assumed to always succeed, unless there is
       an issue in the environment like the loss of network connectivity. That
       will make the method always raise an instance of a subclass of
       :class:`DevlibTransientError` when the command fails, instead of a
       :class:`DevlibStableError`.

.. method:: background(self, command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, as_root=False)

   Execute the command on the connected device, invoking it via subprocess on the host.
   This will return :class:`subprocess.Popen` instance for the command.

   :param command: The command to be executed.
   :param stdout: By default, standard output will be piped from the subprocess;
      this may be used to redirect it to an alternative file handle.
   :param stderr: By default, standard error will be piped from the subprocess;
      this may be used to redirect it to an alternative file handle.
   :param as_root: The command will be executed as root. This will fail on
       unrooted connected devices.

   .. note:: This **will block the connection** until the command completes.

.. note:: The above methods are directly wrapped by :class:`~devlib.target.Target` methods,
          however note that some of the defaults are different.

.. method:: cancel_running_command(self)

   Cancel a running command (previously started with :func:`background`) and free up the connection.
   It is valid to call this if the command has already terminated (or if no
   command was issued), in which case this is a no-op.

.. method:: close(self)

   Close the connection to the device. The :class:`Connection` object should not
   be used after this method is called. There is no way to reopen a previously
   closed connection, a new connection object should be created instead.

.. note:: There is no :func:`open` method, as the connection is assumed to be
          opened on instantiation.


.. _connection-types:

Connection Types
----------------


.. module:: devlib.utils.android

.. class:: AdbConnection(device=None, timeout=None, adb_server=None, adb_as_root=False, connection_attempts=MAX_ATTEMPTS,\
                         poll_transfers=False, start_transfer_poll_delay=30, total_transfer_timeout=3600,\
                         transfer_poll_period=30)

    A connection to an android device via ``adb`` (Android Debug Bridge).
    ``adb`` is part of the Android SDK (though stand-alone versions are also
    available).

    :param device: The name of the adb device. This is usually a unique hex
                   string for USB-connected devices, or an ip address/port
                   combination. To see connected devices, you can run ``adb
                   devices`` on the host.
    :param timeout: Connection timeout in seconds. If a connection to the device
                    is not established within this period, :class:`HostError`
                    is raised.
    :param adb_server: Allows specifying the address of the adb server to use.
    :param adb_as_root: Specify whether the adb server should be restarted in root mode.
    :param connection_attempts: Specify how many connection attempts, 10 seconds
                                apart, should be attempted to connect to the device.
                                Defaults to 5.
    :param poll_transfers: Specify whether file transfers should be polled. Polling
                           monitors the progress of file transfers and periodically
                           checks whether they have stalled, attempting to cancel
                           the transfers prematurely if so.
    :param start_transfer_poll_delay: If transfers are polled, specify the length of
                                      time after a transfer has started before polling
                                      should start.
    :param total_transfer_timeout: If transfers are polled, specify the total amount of time
                                   to elapse before the transfer is cancelled, regardless
                                   of its activity.
    :param transfer_poll_period: If transfers are polled, specify the period at which
                                 the transfers are sampled for activity. Too small values
                                 may cause the destination size to appear the same over
                                 one or more sample periods, causing improper transfer
                                 cancellation.



.. module:: devlib.utils.ssh

.. class:: SshConnection(host, username, password=None, keyfile=None, port=22,\
                         timeout=None, platform=None, \
                         sudo_cmd="sudo -- sh -c {}", strict_host_check=True, \
                         use_scp=False, poll_transfers=False, \
                         start_transfer_poll_delay=30, total_transfer_timeout=3600,\
                         transfer_poll_period=30)

    A connection to a device on the network over SSH.

    :param host: SSH host to which to connect
    :param username: username for SSH login
    :param password: password for the SSH connection

                     .. note:: To connect to a system without a password this
                               parameter should be set to an empty string otherwise
                               ssh key authentication will be attempted.
                     .. note:: In order to user password-based authentication,
                               ``sshpass`` utility must be installed on the
                               system.

    :param keyfile: Path to the SSH private key to be used for the connection.

                    .. note:: ``keyfile`` and ``password`` can't be specified
                              at the same time.

    :param port: TCP port on which SSH server is listening on the remote device.
                 Omit to use the default port.
    :param timeout: Timeout for the connection in seconds. If a connection
                    cannot be established within this time, an error will be
                    raised.
    :param platform: Specify the platform to be used. The generic :class:`~devlib.platform.Platform`
                     class is used by default.
    :param sudo_cmd: Specify the format of the command used to grant sudo access.
    :param strict_host_check: Specify the ssh connection parameter
			     ``StrictHostKeyChecking``. If a path is passed
			     rather than a boolean, it will be taken for a
			     ``known_hosts`` file.  Otherwise, the default
			     ``$HOME/.ssh/known_hosts`` will be used.
    :param use_scp: Use SCP for file transfers, defaults to SFTP.
    :param poll_transfers: Specify whether file transfers should be polled. Polling
                           monitors the progress of file transfers and periodically
                           checks whether they have stalled, attempting to cancel
                           the transfers prematurely if so.
    :param start_transfer_poll_delay: If transfers are polled, specify the length of
                                      time after a transfer has started before polling
                                      should start.
    :param total_transfer_timeout: If transfers are polled, specify the total amount of time
                                   to elapse before the transfer is cancelled, regardless
                                   of its activity.
    :param transfer_poll_period: If transfers are polled, specify the period at which
                                 the transfers are sampled for activity. Too small values
                                 may cause the destination size to appear the same over
                                 one or more sample periods, causing improper transfer
                                 cancellation.

.. class:: TelnetConnection(host, username, password=None, port=None,\
                            timeout=None, password_prompt=None,\
                            original_prompt=None)

    A connection to a device on the network over Telnet.

    .. note:: Since Telnet protocol is does not support file transfer, scp is
              used for that purpose.

    :param host: SSH host to which to connect
    :param username: username for SSH login
    :param password: password for the SSH connection

                     .. note:: In order to user password-based authentication,
                               ``sshpass`` utility must be installed on the
                               system.

    :param port: TCP port on which SSH server is listening on the remote device.
                 Omit to use the default port.
    :param timeout: Timeout for the connection in seconds. If a connection
                    cannot be established within this time, an error will be
                    raised.
    :param password_prompt: A string with the password prompt used by
                            ``sshpass``. Set this if your version of ``sshpass``
                            uses something other than ``"[sudo] password"``.
    :param original_prompt: A regex for the shell prompted presented in the Telnet
                            connection (the prompt will be reset to a
                            randomly-generated pattern for the duration of the
                            connection to reduce the possibility of clashes).
                            This parameter is ignored for SSH connections.

.. module:: devlib.host

.. class:: LocalConnection(keep_password=True, unrooted=False, password=None)

    A connection to the local host allowing it to be treated as a Target.


    :param keep_password: If this is ``True`` (the default) user's password will
                          be cached in memory after it is first requested.
    :param unrooted: If set to ``True``, the platform will be assumed to be
                     unrooted without testing for root. This is useful to avoid
                     blocking on password request in scripts.
    :param password: Specify password on connection creation rather than
                     prompting for it.


.. module:: devlib.utils.ssh
   :noindex:

.. class:: Gem5Connection(platform, host=None, username=None, password=None,\
                          timeout=None, password_prompt=None,\
                          original_prompt=None)

    A connection to a gem5 simulation using a local Telnet connection.

    .. note:: Some of the following input parameters are optional and will be ignored during
              initialisation. They were kept to keep the analogy with a :class:`TelnetConnection`
              (i.e. ``host``, ``username``, ``password``, ``port``,
              ``password_prompt`` and ``original_promp``)


    :param host: Host on which the gem5 simulation is running

                     .. note:: Even though the input parameter for the ``host``
                               will be ignored, the gem5 simulation needs to be
                               on the same host the user is currently on, so if
                               the host given as input parameter is not the
                               same as the actual host, a :class:`TargetStableError`
                               will be raised to prevent confusion.

    :param username: Username in the simulated system
    :param password: No password required in gem5 so does not need to be set
    :param port: Telnet port to connect to gem5. This does not need to be set
                 at initialisation as this will either be determined by the
                 :class:`Gem5SimulationPlatform` or can be set using the
                 :func:`connect_gem5` method
    :param timeout: Timeout for the connection in seconds. Gem5 has high
                    latencies so unless the timeout given by the user via
                    this input parameter is higher than the default one
                    (3600 seconds), this input parameter will be ignored.
    :param password_prompt: A string with password prompt
    :param original_prompt: A regex for the shell prompt

There are two classes that inherit from :class:`Gem5Connection`:
:class:`AndroidGem5Connection` and :class:`LinuxGem5Connection`.
They inherit *almost* all methods from the parent class, without altering them.
The only methods discussed below are those that will be overwritten by the
:class:`LinuxGem5Connection` and :class:`AndroidGem5Connection` respectively.

.. class:: LinuxGem5Connection

    A connection to a gem5 simulation that emulates a Linux system.

    .. method:: _login_to_device(self)

        Login to the gem5 simulated system.

.. class:: AndroidGem5Connection

    A connection to a gem5 simulation that emulates an Android system.

    .. method:: _wait_for_boot(self)

        Wait for the gem5 simulated system to have booted and finished the booting animation.
