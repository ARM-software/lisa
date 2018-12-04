Overview
========

A :class:`Target` instance serves as the main interface to the target device.
There are currently four target interfaces:

- :class:`LinuxTarget` for interacting with Linux devices over SSH.
- :class:`AndroidTarget` for interacting with Android devices over adb.
- :class:`ChromeOsTarget`: for interacting with ChromeOS devices over SSH, and
                           their Android containers over adb.
- :class:`LocalLinuxTarget`: for interacting with the local Linux host.

They all work in more-or-less the same way, with the major difference being in
how connection settings are specified; though there may also be a few APIs
specific to a particular target type (e.g. :class:`AndroidTarget` exposes
methods for working with logcat).


Acquiring a Target
------------------

To create an interface to your device, you just need to instantiate one of the
:class:`Target` derivatives listed above, and pass it the right
``connection_settings``. Code snippet below gives a typical example of
instantiating each of the three target types.

.. code:: python

   from devlib import LocalLinuxTarget, LinuxTarget, AndroidTarget

   # Local machine requires no special connection settings.
   t1 = LocalLinuxTarget()

   # For a Linux device, you will need to provide the normal SSH credentials.
   # Both password-based, and key-based authentication is supported (password
   # authentication requires sshpass to be installed on your host machine).'
   t2 = LinuxTarget(connection_settings={'host': '192.168.0.5',
                                        'username': 'root',
                                        'password': 'sekrit',
                                        # or
                                        'keyfile': '/home/me/.ssh/id_rsa'})
   # ChromeOsTarget connection is performed in the same way as LinuxTarget

   # For an Android target, you will need to pass the device name as reported
   # by "adb devices". If there is only one device visible to adb, you can omit
   # this setting and instantiate similar to a local target.
   t3 = AndroidTarget(connection_settings={'device': '0123456789abcde'})

Instantiating a target may take a second or two as the remote device will be
queried to initialize :class:`Target`'s internal state. If you would like to
create a :class:`Target` instance but not immediately connect to the remote
device, you can pass ``connect=False`` parameter. If you do that, you would have
to then explicitly call ``t.connect()`` before you can interact with the device.

There are a few additional parameters you can pass in instantiation besides
``connection_settings``, but they are usually unnecessary. Please see
:class:`Target` API documentation for more details.

Target Interface
----------------

This is a quick overview of the basic interface to the device. See
:class:`Target` API documentation for the full list of supported methods and
more detailed documentation.

One-time Setup
~~~~~~~~~~~~~~

.. code:: python

   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   t.setup()

This sets up the target for ``devlib`` interaction. This includes creating
working directories, deploying busybox, etc. It's usually enough to do this once
for a new device, as the changes this makes will persist across reboots.
However, there is no issue with calling this multiple times, so, to be on the
safe side, it's a good idea to call this once at the beginning of your scripts.

Command Execution
~~~~~~~~~~~~~~~~~

There are several ways to execute a command on the target. In each case, an
instance of a subclass of :class:`TargetError` will be raised if something goes
wrong. When a transient error is encountered such as the loss of the network
connectivity, it will raise a :class:`TargetTransientError`. When the command
fails, it will raise a :class:`TargetStableError` unless the
``will_succeed=True`` parameter is specified, in which case a
:class:`TargetTransientError` will be raised since it is assumed that the
command cannot fail unless there is an environment issue. In each case, it is
also possible to specify ``as_root=True`` if the specified command should be
executed as root.

.. code:: python

   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   # Execute a command
   output = t.execute('echo $PWD')

   # Execute command via a subprocess and return the corresponding Popen object.
   # This will block current connection to the device until the command
   # completes.
   p = t.background('echo $PWD')
   output, error = p.communicate()

   # Run the command in the background on the device and return immediately.
   # This will not block the connection, allowing to immediately execute another
   # command.
   t.kick_off('echo $PWD')

   # This is used to invoke an executable binary on the device. This allows some
   # finer-grained control over the invocation, such as specifying the directory
   # in which the executable will run; however you're limited to a single binary
   # and cannot construct complex commands (e.g. this does not allow chaining or
   # piping several commands together).
   output = t.invoke('echo', args=['$PWD'], in_directory='/')

File Transfer
~~~~~~~~~~~~~

.. code:: python

   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   # "push" a file from the local machine onto the target device.
   t.push('/path/to/local/file.txt', '/path/to/target/file.txt')

   # "pull" a file from the target device into a location on the local machine
   t.pull('/path/to/target/file.txt', '/path/to/local/file.txt')

   # Install the specified binary on the target. This will deploy the file and
   # ensure it's executable. This will *not* guarantee that the binary will be
   # in PATH. Instead the path to the binary will be returned; this should be
   # used to call the binary henceforth.
   target_bin = t.install('/path/to/local/bin.exe')
   # Example invocation:
   output = t.execute('{} --some-option'.format(target_bin))

The usual access permission constraints on the user account (both on the target
and the host) apply.

Process Control
~~~~~~~~~~~~~~~

.. code:: python

   import signal
   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   # return PIDs of all running instances of a process
   pids = t.get_pids_of('sshd')

   # kill a running process. This works the same ways as the kill command, so
   # SIGTERM will be used by default.
   t.kill(666, signal=signal.SIGKILL)

   # kill all running instances of a process.
   t.killall('badexe', signal=signal.SIGKILL)

   # List processes running on the target. This returns a list of parsed
   # PsEntry records.
   entries = t.ps()
   # e.g.  print virtual memory sizes of all running sshd processes:
   print ', '.join(str(e.vsize) for e in entries if e.name == 'sshd')


More...
~~~~~~~

As mentioned previously, the above is not intended to be exhaustive
documentation of the :class:`Target` interface. Please refer to the API
documentation for the full list of attributes and methods and their parameters.

Super User Privileges
---------------------

It is not necessary for the account logged in on the target to have super user
privileges, however the functionality will obviously be diminished, if that is
not the case. ``devlib`` will determine if the logged in user has root
privileges and the correct way to invoke it. You should avoid including "sudo"
directly in your commands, instead, specify ``as_root=True`` where needed. This
will make your scripts portable across multiple devices and OS's.


On-Target Locations
-------------------

File system layouts vary wildly between devices and operating systems.
Hard-coding absolute paths in your scripts will mean there is a good chance they
will break if run on a different device.  To help with this, ``devlib`` defines
a couple of "standard" locations and a means of working with them.

working_directory
        This is a directory on the target readable and writable by the account
        used to log in. This should generally be used for all output generated
        by your script on the device and as the destination for all
        host-to-target file transfers. It may or may not permit execution so
        executables should not be run directly from here.

executables_directory
        This directory allows execution. This will be used by ``install()``.

.. code:: python

   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   # t.path  is equivalent to Python standard library's os.path, and should be
   # used in the same way. This insures that your scripts are portable across
   # both target and host OS variations. e.g.
   on_target_path = t.path.join(t.working_directory, 'assets.tar.gz')
   t.push('/local/path/to/assets.tar.gz', on_target_path)

   # Since working_directory is a common base path for on-target locations,
   # there a short-hand for the above:
   t.push('/local/path/to/assets.tar.gz', t.get_workpath('assets.tar.gz'))


Exceptions Handling
-------------------

Devlib custom exceptions all derive from :class:`DevlibError`. Some exceptions
are further categorized into :class:`DevlibTransientError` and
:class:`DevlibStableError`. Transient errors are raised when there is an issue
in the environment that can happen randomly such as the loss of network
connectivity. Even a properly configured environment can be subject to such
transient errors. Stable errors are related to either programming errors or
configuration issues in the broad sense. This distinction allows quicker
analysis of failures, since most transient errors can be ignored unless they
happen at an alarming rate. :class:`DevlibTransientError` usually propagates up
to the caller of devlib APIs, since it means that an operation could not
complete. Retrying it or bailing out is therefore a responsability of the caller.

The hierarchy is as follows:

- :class:`DevlibError`
   
   - :class:`WorkerThreadError`
   - :class:`HostError`
   - :class:`TargetError`
      
      - :class:`TargetStableError`
      - :class:`TargetTransientError`
      - :class:`TargetNotRespondingError`
   
   - :class:`DevlibStableError`
      
      - :class:`TargetStableError`

   - :class:`DevlibTransientError`

      - :class:`TimeoutError`
      - :class:`TargetTransientError`
      - :class:`TargetNotRespondingError`


Extending devlib
~~~~~~~~~~~~~~~~

New devlib code is likely to face the decision of raising a transient or stable
error. When it is unclear which one should be used, it can generally be assumed
that the system is properly configured and therefore, the error is linked to an
environment transient failure. If a function is somehow probing a property of a
system in the broad meaning, it can use a stable error as a way to signal a
non-expected value of that property even if it can also face transient errors.
An example are the various ``execute()`` methods where the command can generally
not be assumed to be supposed to succeed by devlib. Their failure does not
usually come from an environment random issue, but for example a permission
error. The user can use such expected failure to probe the system. Another
example is boot completion detection on Android: boot failure cannot be
distinguished from a timeout which is too small. A non-transient exception is
still raised, since assuming the timeout comes from a network failure would
either make the function useless, or force the calling code to handle a
transient exception under normal operation. The calling code would potentially
wrongly catch transient exceptions raised by other functions as well and attach
a wrong meaning to them.


Modules
-------

Additional functionality is exposed via modules. Modules are initialized as
attributes of a target instance. By default, ``hotplug``, ``cpufreq``,
``cpuidle``, ``cgroups`` and ``hwmon`` will attempt to load on target; additional
modules may be specified when creating a :class:`Target` instance.

A module will probe the target for support before attempting to load. So if the
underlying platform does not support particular functionality (e.g. the kernel
on target device was built without hotplug support). To check whether a module
has been successfully installed on a target, you can use ``has()`` method, e.g.

.. code:: python

   from devlib import LocalLinuxTarget
   t = LocalLinuxTarget()

   cpu0_freqs = []
   if t.has('cpufreq'):
       cpu0_freqs = t.cpufreq.list_frequencies(0)


Please see the modules documentation for more detail.


Measurement and Trace
---------------------

You can collected traces (currently, just ftrace) using
:class:`TraceCollector`\ s. For example

.. code:: python

   from devlib import AndroidTarget, FtraceCollector
   t = LocalLinuxTarget()

   # Initialize a collector specifying the events you want to collect and
   # the buffer size to be used.
   trace = FtraceCollector(t, events=['power*'], buffer_size=40000)

   # As a context manager, clear ftrace buffer using trace.reset(),
   # start trace collection using trace.start(), then stop it Using
   # trace.stop(). Using a context manager brings the guarantee that
   # tracing will stop even if an exception occurs, including 
   # KeyboardInterrupt (ctr-C) and SystemExit (sys.exit)
   with trace:
      # Perform the operations you want to trace here...
      import time; time.sleep(5)

   # extract the trace file from the target into a local file
   trace.get_trace('/tmp/trace.bin')

   # View trace file using Kernelshark (must be installed on the host).
   trace.view('/tmp/trace.bin')

   # Convert binary trace into text format. This would normally be done
   # automatically during get_trace(), unless autoreport is set to False during
   # instantiation of the trace collector.
   trace.report('/tmp/trace.bin', '/tmp/trace.txt')

In a similar way, :class:`Instrument` instances may be used to collect
measurements (such as power) from targets that support it. Please see
instruments documentation for more details.
