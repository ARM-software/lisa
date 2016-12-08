Connection
==========

A :class:`Connection` abstracts an actual physical connection to a device. The
first connection is created when :func:`Target.connect` method is called. If a
:class:`Target` is used in a multi-threaded environment, it will maintain a
connection for each thread in which it is invoked. This allows the same target
object to be used in parallel in multiple threads.

:class:`Connection`\ s will be automatically created and managed by
:class:`Target`\ s, so there is usually no reason to create one manually.
Instead, configuration for a :class:`Connection` is passed as
`connection_settings` parameter when creating a :class:`Target`. The connection
to be used target is also specified on instantiation by `conn_cls` parameter,
though all concrete :class:`Target` implementations will set an appropriate
default, so there is typically no need to specify this explicitly.

:class:`Connection` classes are not a part of an inheritance hierarchy, i.e.
they do not derive from a common base. Instead, a :class:`Connection` is any
class that implements the following methods.


.. method:: push(self, source, dest, timeout=None)

   Transfer a file from the host machine to the connected device.

   :param source: path of to the file on the host
   :param dest: path of to the file on the connected
   :param timeout: timeout (in seconds) for the transfer; if the transfer does
       not  complete within this period, an exception will be raised.

.. method:: pull(self, source, dest, timeout=None)

   Transfer a file from the connected device to the host machine.

   :param source: path of to the file on the connected device
   :param dest: path of to the file on the host
   :param timeout: timeout (in seconds) for the transfer; if the transfer does
       not  complete within this period, an exception will be raised.

.. method:: execute(self, command, timeout=None, check_exit_code=False, as_root=False)

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
   
.. note:: The above methods are directly wrapped by :class:`Target` methods,
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

.. class:: AdbConnection(device=None, timeout=None)

    A connection to an android device via ``adb`` (Android Debug Bridge).
    ``adb`` is part of the Android SDK (though stand-alone versions are also
    available).

    :param device: The name of the adb divice. This is usually a unique hex
                   string for USB-connected devices, or an ip address/port
                   combination. To see connected devices, you can run ``adb
                   devices`` on the host.
    :param timeout: Connection timeout in seconds. If a connection to the device
                    is not esblished within this period, :class:`HostError` 
                    is raised.


.. class:: SshConnection(host, username, password=None, keyfile=None, port=None,\
                         timeout=None, password_prompt=None)

    A connectioned to a device on the network over SSH.

    :param host: SSH host to which to connect
    :param username: username for SSH login
    :param password: password for the SSH connection

                     .. note:: In order to user password-based authentication,
                               ``sshpass`` utility must be installed on the
                               system.
    
    :param keyfile: Path to the SSH private key to be used for the connection.

                    .. note:: ``keyfile`` and ``password`` can't be specified 
                              at the same time.

    :param port: TCP port on which SSH server is litening on the remoted device.
                 Omit to use the default port.
    :param timeout: Timeout for the connection in seconds. If a connection
                    cannot be established within this time, an error will be
                    raised.
    :param password_prompt: A string with the password prompt used by
                            ``sshpass``. Set this if your version of ``sshpass``
                            uses somethin other than ``"[sudo] password"``.


.. class:: TelnetConnection(host, username, password=None, port=None,\
                            timeout=None, password_prompt=None,\
                            original_prompt=None)

    A connectioned to a device on the network over Telenet.

    .. note:: Since Telenet protocol is does not support file transfer, scp is
              used for that purpose.

    :param host: SSH host to which to connect
    :param username: username for SSH login
    :param password: password for the SSH connection

                     .. note:: In order to user password-based authentication,
                               ``sshpass`` utility must be installed on the
                               system.

    :param port: TCP port on which SSH server is litening on the remoted device.
                 Omit to use the default port.
    :param timeout: Timeout for the connection in seconds. If a connection
                    cannot be established within this time, an error will be
                    raised.
    :param password_prompt: A string with the password prompt used by
                            ``sshpass``. Set this if your version of ``sshpass``
                            uses somethin other than ``"[sudo] password"``.
    :param original_prompt: A regex for the shell prompted presented in the Telenet
                            connection (the prompt will be reset to a
                            randomly-generated pattern for the duration of the
                            connection to reduce the possibility of clashes).
                            This paramer is ignored for SSH connections.


.. class:: LocalConnection(keep_password=True, unrooted=False, password=None)

    A connection to the local host allowing it to be treated as a Target.


    :param keep_password: If this is ``True`` (the default) user's password will
                          be cached in memory after it is first requested. 
    :param unrooted: If set to ``True``, the platform will be assumed to be
                     unrooted without testing for root. This is useful to avoid
                     blocking on password request in scripts.
    :param password: Specify password on connection creation rather than
                     prompting for it.
