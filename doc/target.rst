Target
======


.. class:: Target(connection_settings=None, platform=None, working_directory=None, executables_directory=None, connect=True, modules=None, load_default_modules=True, shell_prompt=DEFAULT_SHELL_PROMPT)
   
    :class:`Target` is the primary interface to the remote device. All interactions
    with the device are performed via a :class:`Target` instance, either
    directly, or via its modules or a wrapper interface (such as an
    :class:`Instrument`).

    :param connection_settings: A ``dict`` that specifies how to connect to the remote 
       device. Its contents depend on the specific :class:`Target` type (used see
       :ref:`connection-types`\ ).

    :param platform: A :class:`Target` defines interactions at Operating System level. A 
        :class:`Platform` describes the underlying hardware (such as CPUs
        available). If a :class:`Platform` instance is not specified on
        :class:`Target` creation, one will be created automatically and it will
        dynamically probe the device to discover as much about the underlying
        hardware as it can. See also :ref:`platform`\ .

    :param working_directory: This is primary location for on-target file system
        interactions performed by ``devlib``. This location *must* be readable and
        writable directly (i.e. without sudo) by the connection's user account. 
        It may or may not allow execution. This location will be created, 
        if necessary, during ``setup()``.

        If not explicitly specified, this will be set to a default value
        depending on the type of :class:`Target`

    :param executables_directory: This is the location to which ``devlib`` will
        install executable binaries (either during ``setup()`` or via an
        explicit ``install()`` call). This location *must* support execution
        (obviously). It should also be possible to write to this location,
        possibly with elevated privileges (i.e. on a rooted Linux target, it
        should be possible to write here with sudo, but not necessarily directly
        by the connection's account). This location will be created, 
        if necessary, during ``setup()``.

        This location does *not* to be same as the system's executables
        location. In fact, to prevent devlib from overwriting system's defaults,
        it better if this is a separate location, if possible.

        If not explicitly specified, this will be set to a default value
        depending on the type of :class:`Target`

    :param connect: Specifies whether a connections should be established to the
        target. If this is set to ``False``, then ``connect()`` must be
        explicitly called later on before the :class:`Target` instance can be
        used.

    :param modules: a list of additional modules to be installed. Some modules will
        try to install by default (if supported by the underlying target).
        Current default modules are ``hotplug``, ``cpufreq``, ``cpuidle``, 
        ``cgroups``, and ``hwmon`` (See :ref:`modules`\ ).

        See modules documentation for more detail.

    :param load_default_modules: If set to ``False``,  default modules listed
         above will *not* attempt to load. This may be used to either speed up
         target instantiation (probing for initializing modules takes a bit of time)
         or if there is an issue with one of the modules on a particular device
         (the rest of the modules will then have to be explicitly specified in
         the ``modules``).

    :param shell_prompt: This is a regular expression that matches the shell
         prompted on the target. This may be used by some modules that establish
         auxiliary connections to a target over UART.

.. attribute:: Target.core_names

   This is a list containing names of CPU cores on the target, in the order in
   which they are index by the kernel. This is obtained via the underlying
   :class:`Platform`.

.. attribute:: Target.core_clusters

   Some devices feature heterogeneous core configurations (such as ARM
   big.LITTLE).  This is a list that maps CPUs onto underlying clusters.
   (Usually, but not always, clusters correspond to groups of CPUs with the same
   name). This is obtained via the underlying :class:`Platform`.

.. attribute:: Target.big_core

   This is the name of the cores that the "big"s in an ARM big.LITTLE
   configuration. This is obtained via the underlying :class:`Platform`.

.. attribute:: Target.little_core

   This is the name of the cores that the "little"s in an ARM big.LITTLE
   configuration. This is obtained via the underlying :class:`Platform`.

.. attribute:: Target.is_connected

   A boolean value that indicates whether an active connection exists to the
   target device. 

.. attribute:: Target.connected_as_root

   A boolean value that indicate whether the account that was used to connect to
   the target device is "root" (uid=0).

.. attribute:: Target.is_rooted

   A boolean value that indicates whether the connected user has super user
   privileges on the devices (either is root, or is a sudoer).

.. attribute:: Target.kernel_version

   The version of the kernel on the target device. This returns a
   :class:`KernelVersion` instance that has separate ``version`` and ``release``
   fields.

.. attribute:: Target.os_version

   This is a dict that contains a mapping of OS version elements to their
   values. This mapping is OS-specific.

.. attribute:: Target.cpuinfo

   This is a :class:`Cpuinfo` instance which contains parsed contents of
   ``/proc/cpuinfo``.

.. attribute:: Target.number_of_cpus

   The total number of CPU cores on the target device.

.. attribute:: Target.config

   A :class:`KernelConfig` instance that contains parsed kernel config from the
   target device. This may be ``None`` if kernel config could not be extracted.

.. attribute:: Target.user

   The name of the user logged in on the target device.

.. attribute:: Target.conn

   The underlying connection object. This will be ``None`` if an active
   connection does not exist (e.g. if ``connect=False`` as passed on
   initialization and ``connect()`` has not been called).

   .. note:: a :class:`Target` will automatically create a connection per
             thread. This will always be set to the connection for the current
             thread.

.. method:: Target.connect([timeout])
   
   Establish a connection to the target. It is usually not necessary to call
   this explicitly, as a connection gets automatically established on
   instantiation.

.. method:: Target.disconnect()

   Disconnect from target, closing all active connections to it.

.. method:: Target.get_connection([timeout])

   Get an additional connection to the target. A connection can be used to
   execute one blocking command at time. This will return a connection that can
   be used to interact with a target in parallel while a blocking operation is
   being executed.

   This should *not* be used to establish an initial connection; use
   ``connect()`` instead.

   .. note:: :class:`Target` will automatically create a connection per
             thread, so you don't normally need to use this explicitly in
             threaded code. This is generally useful if you want to perform a
             blocking operation (e.g. using ``background()``) while at the same
             time doing something else in the same host-side thread.

.. method:: Target.setup([executables])

   This will perform an initial one-time set up of a device for devlib
   interaction. This involves deployment of tools relied on the :class:`Target`,
   creation of working locations on the device, etc.

   Usually, it is enough to call this method once per new device, as its effects
   will persist across reboots. However, it is safe to call this method multiple
   times. It may therefore be a good practice to always call it once at the
   beginning of a script to ensure that subsequent interactions will succeed.

   Optionally, this may also be used to deploy additional tools to the device
   by specifying a list of binaries to install in the ``executables`` parameter.

.. method:: Target.reboot([hard [, connect, [timeout]]])

   Reboot the target device.

   :param hard: A boolean value. If ``True`` a hard reset will be used instead
        of the usual soft reset. Hard reset must be supported (usually via a
        module) for this to work. Defaults to ``False``.
   :param connect: A boolean value. If ``True``, a connection will be
        automatically established to the target after reboot. Defaults to
        ``True``.
   :param timeout: If set, this will be used by various (platform-specific)
        operations during reboot process to detect if the reboot has failed and
        the device has hung.

.. method:: Target.push(source, dest [, timeout])

   Transfer a file from the host machine to the target device.

   :param source: path of to the file on the host
   :param dest: path of to the file on the target
   :param timeout: timeout (in seconds) for the transfer; if the transfer does
       not  complete within this period, an exception will be raised.

.. method:: Target.pull(source, dest [, timeout])

   Transfer a file from the target device to the host machine.

   :param source: path of to the file on the target
   :param dest: path of to the file on the host
   :param timeout: timeout (in seconds) for the transfer; if the transfer does
       not  complete within this period, an exception will be raised.

.. method:: Target.execute(command [, timeout [, check_exit_code [, as_root]]])

   Execute the specified command on the target device and return its output.

   :param command: The command to be executed.
   :param timeout: Timeout (in seconds) for the execution of the command. If
       specified, an exception will be raised if execution does not complete
       with the specified period.
   :param check_exit_code: If ``True`` (the default) the exit code (on target) 
       from execution of the command will be checked, and an exception will be
       raised if it is not ``0``.
   :param as_root: The command will be executed as root. This will fail on
       unrooted targets.

.. method:: Target.background(command [, stdout [, stderr [, as_root]]])

   Execute the command on the target, invoking it via subprocess on the host.
   This will return :class:`subprocess.Popen` instance for the command.

   :param command: The command to be executed.
   :param stdout: By default, standard output will be piped from the subprocess;
      this may be used to redirect it to an alternative file handle.
   :param stderr: By default, standard error will be piped from the subprocess;
      this may be used to redirect it to an alternative file handle.
   :param as_root: The command will be executed as root. This will fail on
       unrooted targets.

   .. note:: This **will block the connection** until the command completes.

.. method:: Target.invoke(binary [, args [, in_directory [, on_cpus [, as_root [, timeout]]]]])

   Execute the specified binary on target (must already be installed) under the
   specified conditions and return the output.

   :param binary: binary to execute. Must be present and executable on the device.
   :param args: arguments to be passed to the binary. The can be either a list or
          a string.
   :param in_directory:  execute the binary in the  specified directory. This must
                   be an absolute path.
   :param on_cpus:  taskset the binary to these CPUs. This may be a single ``int`` (in which
          case, it will be interpreted as the mask), a list of ``ints``, in which
          case this will be interpreted as the list of cpus, or string, which
          will be interpreted as a comma-separated list of cpu ranges, e.g.
          ``"0,4-7"``.
   :param as_root: Specify whether the command should be run as root
   :param timeout: If this is specified and invocation does not terminate within this number 
           of seconds, an exception will be raised.

.. method:: Target.kick_off(command [, as_root])

   Kick off the specified command on the target and return immediately. Unlike
   ``background()`` this will not block the connection; on the other hand, there
   is not way to know when the command finishes (apart from calling ``ps()``)
   or to get its output (unless its redirected into a file that can be pulled
   later as part of the command).

   :param command: The command to be executed.
   :param as_root: The command will be executed as root. This will fail on
       unrooted targets.

.. method:: Target.read_value(path [,kind])

   Read the value from the specified path. This is primarily intended for
   sysfs/procfs/debugfs etc.

   :param path: file to read
   :param kind: Optionally, read value will be converted into the specified
        kind (which should be a callable that takes exactly one parameter).

.. method:: Target.read_int(self, path)

   Equivalent to ``Target.read_value(path, kind=devlab.utils.types.integer)``

.. method:: Target.read_bool(self, path)

   Equivalent to ``Target.read_value(path, kind=devlab.utils.types.boolean)``

.. method:: Target.write_value(path, value [, verify])

   Write the value to the specified path on the target. This is primarily 
   intended for sysfs/procfs/debugfs etc.

   :param path: file to write into
   :param value: value to be written
   :param verify: If ``True`` (the default) the value will be read back after
       it is written to make sure it has been written successfully. This due to 
       some sysfs entries silently failing to set the written value without
       returning an error code.

.. method:: Target.reset()

   Soft reset the target. Typically, this means executing ``reboot`` on the
   target.

.. method:: Target.check_responsive()

   Returns ``True`` if the target appears to be responsive and ``False``
   otherwise.

.. method:: Target.kill(pid[, signal[, as_root]])

   Kill a process on the target.

   :param pid: PID of the process to be killed.
   :param signal: Signal to be used to kill the process. Defaults to
       ``signal.SIGTERM``.
   :param as_root: If set to ``True``, kill will be issued as root. This will
       fail on unrooted targets.

.. method:: Target.killall(name[, signal[, as_root]])

   Kill all processes with the specified name on the target. Other parameters
   are the same as for ``kill()``.

.. method:: Target.get_pids_of(name)

   Return a list of PIDs of all running instances of the specified process.

.. method:: Target.ps()

   Return a list of :class:`PsEntry` instances for all running processes on the
   system.

.. method:: Target.file_exists(self, filepath)

   Returns ``True`` if the specified path exists on the target and ``False``
   otherwise.

.. method:: Target.list_file_systems()

   Lists file systems mounted on the target. Returns a list of
   :class:`FstabEntry`\ s.

.. method:: Target.list_directory(path[, as_root])

   List (optionally, as root) the contents of the specified directory. Returns a
   list of strings.


.. method:: Target.get_workpath(self, path)

   Convert the specified path to an absolute path relative to
   ``working_directory`` on the target. This is a shortcut for
   ``t.path.join(t.working_directory, path)``

.. method:: Target.tempfile([prefix [, suffix]])

   Get a path to a temporary file (optionally, with the specified prefix and/or
   suffix) on the target.

.. method:: Target.remove(path[, as_root])

   Delete the specified path on the target. Will work on files and directories.

.. method:: Target.core_cpus(core)

   Return a list of numeric cpu IDs corresponding to the specified core name.

.. method:: Target.list_online_cpus([core])

   Return a list of numeric cpu IDs for all online CPUs (optionally, only for
   CPUs corresponding to the specified core).

.. method:: Target.list_offline_cpus([core])

   Return a list of numeric cpu IDs for all offline CPUs (optionally, only for
   CPUs corresponding to the specified core).

.. method:: Target.getenv(variable)

   Return the value of the specified environment variable on the device

.. method:: Target.capture_screen(filepath)

   Take a screenshot on the device and save it to the specified file on the
   host. This may not be supported by the target.

.. method:: Target.install(filepath[, timeout[, with_name]])

   Install an executable on the device.

   :param filepath: path to the executable on the host
   :param timeout: Optional timeout (in seconds) for the installation
   :param with_name: This may be used to rename the executable on the target

.. method:: Target.uninstall(name)

   Uninstall the specified executable from the target

.. method:: Target.get_installed(name)

   Return the full installation path on the target for the specified executable,
   or ``None`` if the executable is not installed.

.. method:: Target.which(name)

   Alias for ``get_installed()``

.. method:: Target.is_installed(name)

   Returns ``True`` if an executable with the specified name is installed on the
   target and ``False`` other wise.

.. method:: Target.extract(path, dest=None)

   Extracts the specified archive/file and returns the path to the extrated
   contents. The extraction method is determined based on the file extension.
   ``zip``, ``tar``, ``gzip``, and ``bzip2`` are supported.

   :param dest: Specified an on-target destination directory (which must exist)
                for the extrated contents.

    Returns the path to the extracted contents. In case of files (gzip and
    bzip2), the path to the decompressed file is returned; for archives, the
    path to the directory with the archive's contents is returned.
