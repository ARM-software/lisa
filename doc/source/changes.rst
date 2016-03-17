=================================
What's New in Workload Automation
=================================
-------------
Version 2.4.0
-------------

Additions:
##########

Devices
~~~~~~~~
- ``gem5_linux`` and ``gem5_android``: Interfaces for Gem5 simulation
  environment running Linux and Android respectively.
- ``XE503C1211``: Interface for Samsung XE503C12 Chromebooks.
- ``chromeos_test_image``: Chrome OS test image device. An off the shelf
  device will not work with this device interface.

Instruments
~~~~~~~~~~~~
- ``freq_sweep``: Allows "sweeping" workloads across multiple CPU frequencies.
- ``screenon``: Ensures screen is on, before each iteration, or periodically
  on Android devices.
- ``energy_model``: This instrument can be used to generate an energy model
  for a device based on collected power and performance measurments.
- ``netstats``:  Allows monitoring data sent/received by applications on an
  Android device.

Modules
~~~~~~~
- ``cgroups``: Allows query and manipulation of cgroups controllers on a Linux
  device. Currently, only cpusets controller is implemented.
- ``cpuidle``: Implements cpuidle state discovery, query and manipulation for
  a Linux device. This replaces the more primitive get_cpuidle_states method
  of LinuxDevice.
- ``cpufreq`` has now been split out into a device module

Reasource Getters
~~~~~~~~~~~~~~~~~
- ``http_assets``:  Downloads resources from a web server.

Results Processors
~~~~~~~~~~~~~~~~~~~
- ``ipynb_exporter``: Generates an IPython notebook from a template with the
  results and runs it.
- ``notify``: Displays a desktop notification when a run finishes
  (Linux only).
- ``cpustates``: Processes power ftrace to produce CPU state and parallelism
  stats. There is also a script to invoke this outside of WA.

Workloads
~~~~~~~~~
- ``telemetry``: Executes Google's Telemetery benchmarking framework
- ``hackbench``: Hackbench runs tests on the Linux scheduler
- ``ebizzy``: This workload resembles common web server application workloads.
- ``power_loadtest``: Continuously cycles through a set of browser-based
  activities and monitors battery drain on a device (part of ChromeOS autotest
  suite).
- ``rt-app``: Simulates configurable real-time periodic load.
- ``linpack-cli``:  Command line version of linpack benchmark.
- ``lmbench``: A suite of portable ANSI/C microbenchmarks for UNIX/POSIX.
- ``stream``: Measures memory bandwidth.
- ``iozone``: Runs a series of disk I/O performance tests.
- ``androbench``:  Measures the storage performance of device.
- ``autotest``:  Executes tests from ChromeOS autotest suite.

Framework
~~~~~~~~~
- ``wlauto.utils``:
   - Added ``trace_cmd``, a generic trace-cmd paraser.
   - Added ``UbootMenu``, allows navigating Das U-boot menu over serial.
- ``wlauto.utils.types``:
   - ``caseless_string``: Behaves exactly like a string, except this ignores
     case in comparisons. It does, however, preserve case.
   - ``list_of``: allows dynamic generation of type-safe list types based on
     an existing type.
   - ``arguments``: represents arguments that are passed on a command line to
     an application.
   - ``list-or``: allows dynamic generation of types that accept either a base
     type or a list of base type. Using this ``list_or_integer``,
     ``list_or_number`` and ``list_or_bool`` were also added.
- ``wlauto.core.configuration.WorkloadRunSpec``:
   - ``copy``: Allows making duplicates of ``WorkloadRunSpec``'s
- ``wlatuo.utils.misc``:
   - ``list_to_ranges`` and ``ranges_to_list``: convert between lists of
     integers and corresponding range strings, e.g. between [0,1,2,4] and
     '0-2,4'
   - ``list_to_mask`` and ``mask_to_list``: convert between lists of integers
     and corresponding integer masks, e.g. between [0,1,2,4] and 0x17
- ``wlauto.instrumentation``:
   - ``instrument_is_enabled``: Returns whether or not an instrument is
     enabled for the current job.
- ``wlauto.core.result``:
   - Added "classifiers" field to Metric objects. This is a dict mapping
     classifier names (arbitrary strings) to corresponding values for that
     specific metrics. This is to allow plugins to add plugin-specific
     annotations to metric that could be handled in a generic way (e.g. by
     result processors). They can also be set in agendas.
- Failed jobs will now be automatically retired
- Implemented dynamic device modules that may be loaded automatically on
  device initialization if the device supports them.
- Added support for YAML configs.
- Added ``initialze`` and ``finalize`` methods to workloads.
- ``wlauto.core.ExecutionContext``:
   - Added ``job_status`` property that returns the status of the currently
     running job.

Fixes/Improvements
##################

Devices
~~~~~~~~
- ``tc2``: Workaround for buffer overrun when loading large initrd blob.
- ``juno``:
     - UEFI config can now be specified as a parameter.
     - Adding support for U-Boot booting.
     - No longer auto-disconnects ADB at the end of a run.
     - Added ``actually_disconnect`` to restore old disconnect behaviour
     - Now passes ``video`` command line to Juno kernel to work around a known
       issue where HDMI loses sync with monitors.
     - Fixed flashing.

Instruments
~~~~~~~~~~~
- ``trace_cmd``:
     - Fixed ``buffer_size_file`` for non-Android devices
     - Reduce starting priority.
     - Now handles trace headers and thread names with spaces
- ``energy_probe``: Added ``device_entry`` parameter.
- ``hwmon``:
     - Sensor discovery is now done only at the start of a run.
     - Now prints both before/after and mean temperatures.
- ``daq``:
     - Now reports energy
     - Fixed file descriptor leak
     - ``daq_power.csv`` now matches the order of labels (if specified).
     - Added ``gpio_sync``. When enabled, this wil cause the instrument to
       insert a marker into ftrace, while at the same time setting a GPIO pin
       high.
     - Added ``negative_values`` parameter. which can be used to specify how
       negative values in the samples should be handled.
     - Added ``merge_channels`` parameter. When set DAQ channel will be summed
       together.
     - Workload labels, rather than names, are now used in the "workload"
       column.
- ``cpufreq``:
     - Fixes missing directories problem.
     - Refined the availability check not to rely on the top-level cpu/cpufreq
       directory
     - Now handles non-integer output in ``get_available_frequencies``.
- ``sysfs_extractor``:
     - No longer raises an error when both device and host paths are empty.
     - Fixed pulled files verification.
- ``perf``:
     - Updated binaries.
     - Added option to force install.
     - ``killall`` is now run as root on rooted Android devices.
- ``fps``:
     - now generates detailed FPS traces as well as report average FPS.
     - Updated jank calcluation to only count "large" janks.
     - Now filters out bogus ``actual-present`` times and ignore janks above
       ``PAUSE_LATENCY``
- ``delay``:
     - Added ``fixed_before_start`` parameter.
     - Changed existing ``*_between_specs`` and ``*_between_iterations``
       callbacks to be ``very_slow``
- ``streamline``:
     - Added Linux support
     - ``gatord`` is now only started once at the start of the run.

modules
~~~~~~~
- ``flashing``:
     - Fixed vexpress flashing
     - Added an option to keep UEFI entry

Result Processors
~~~~~~~~~~~~~~~~~
- ``cpustate``:
     - Now generates a timeline csv as well as stats.
     - Adding ID to overall cpustate reports.
- ``csv``: (partial) ``results.csv`` will now be written after each iteration
  rather than at the end of the run.

Workloads
~~~~~~~~~
- ``glb_corporate``: clears logcat to prevent getting results from previous
  run.
- ``sysbench``:
     - Updated sysbench binary to a statically linked verison
     - Added ``file_test_mode parameter`` - this is a mandatory argumet if
       ``test`` is ``"fileio"``.
     - Added ``cmd_params`` parameter to pass options directily to sysbench
       invocation.
     - Removed Android browser launch and shutdown from workload (now runs on
       both Linux and Android).
     - Now works with unrooted devices.
     - Added the ability to run based on time.
     - Added a parameter to taskset to specific core(s).
     - Added ``threads`` parameter to be consistent with dhrystone.
     - Fixed case where default ``timeout`` < ``max_time``.
- ``Dhrystone``:
     - added ``taskset_mask`` parameter to allow pinning to specific cores.
     - Now kills any running instances during setup (also handles CTRL-C).
- ``sysfs_extractor``: Added parameter to explicitly enable/disable tempfs
  caching.
- ``antutu``:
     - Fixed multi-``times`` playback for v5.
     - Updated result parsing to handle Android M logcat output.
- ``geekbench``: Increased timout to cater for slower devices.
- ``idle``: Now works on Linux devices.
- ``manhattan``: Added ``run_timemout`` parameter.
- ``bbench``: Now works when binaries_directory is not in path.
- ``nemamark``: Made duration configurable.

Framework
~~~~~~~~~~
- ``BaseLinuxDevice``:
     - Now checks that at least one core is enabled on another cluster before
       attempting to set number of cores on a cluster to ``0``.
     - No longer uses ``sudo`` if already logged in as ``root``.
     - Now saves ``dumpsys window`` output to the ``__meta`` directory.
     - Now takes ``password_prompt`` as a parameter for devices with a non
       standard ``sudo`` password prompt.
     - No longer raises an error if ``keyfile`` or ``password`` are not
       provided when they are not necessary.
     - Added new cpufreq APIs:
        - ``core`` APIs take a core name as the parameter (e.g. "a15")
        - ``cluster`` APIs take a numeric cluster ID (eg. 0)
        - ``cpu`` APIs take a cpufreq cpu ID as a parameter.
     - ``set_cpu_frequency`` now has a ``exact`` parameter. When true (the
       default) it will produce an error when the specified frequency is not
       supported by the cpu, otherwise cpufreq will decide what to do.
     - Added ``{core}_frequency`` runtime parameter to set cluster frequency.
     - Added ``abi`` property.
     - ``get_properties`` moved from ``LinuxDevice``, meaning ``AndroidDevice``
       will try to pull the same files. Added more paths to pull by default
       too.
     - fixed ``list_file_systems`` for Android M and Linux devices.
     - Now sets ``core_clusters`` from ``core_names`` if not explicitly
       specified.
     - Added ``invoke`` method that allows invoking an executable on the device
       under controlled contions (e.g. within a particular directory, or
       taskset to specific CPUs).
     - No longer attempts to ``get_sysfile_value()`` as root on unrooted
       devices.
- ``LinuxDevice``:
     - Now creates ``binaries_directory`` path if it doesn't exist.
     - Fixed device reset
     - Fixed ``file_exists``
     - implemented ``get_pid_of()`` and ``ps()``. Existing implementation
       relied on Android version of ps.
     - ``listdir`` will now return an empty list for an empty directory
       instead of a list containing a single empty string.
- ``AndroidDevice``:
     - Executable (un)installation now works on unrooted devices.
     - Now takes into account ``binar_directory`` when setting up busybox path.
     - update ``android_prompt`` so that it works even if is not ``"/"``
     - ``adb_connect``: do not assume port 5555 anymore.
     - Now always deploys busybox on rooted devices.
     - Added ``swipe_to_unlock`` method.
- Fixed initialization of ``~/.workload_automation.``.
- Fixed replaying events using revent on 64 bit platforms.
- Improved error repoting when loading plugins.
- ``result`` objects now track their output directories.
- ``context.result`` will not result in ``context.run_result`` when not
  executing a job.
- ``wlauto.utils.ssh``:
     - Fixed key-based authentication.
     - Fixed carriage return stripping in ssh.
     - Now takes ``password_prompt`` as a parameter for non standard ``sudo``
       password prompts.
     - Now with 100% more thread safety!
     - If a timeout condition is hit, ^C is now sent to kill the current
       foreground process and make the shell available for subsequent commands.
     - More robust ``exit_code`` handling for ssh interface
     - Now attempts to deal with dropped connections
     - Fixed error reporting on failed exit code extraction.
     - Now handles backspaces in serial output
     - Added ``port`` argument for telnet connections.
     - Now allows telnet connections without a password.
- Fixed config processing for plugins with non-identifier names.
- Fixed ``get_meansd`` for numbers < 1
- ``wlatuo.utils.ipython``:
     - Now supports old versions of IPython
     - Updated version check to only initialize ipython utils if version is
       < 4.0.0. Version 4.0.0 changes API and breaks WA's usage of it.
- Added ``ignore`` parameter to ``check_output``
- Agendas:
     - Now raise an error if an agenda contains duplicate keys
     - Now raise an error if config section in an agenda is not dict-like
     - Now properly handles ``core_names`` and ``core_clusters``
     - When merging list parameters from different sources, duplicates are no
       longer removed.
- The ``INITIAL_BOOT`` signal is now sent went performing a hard reset during
  intial boot
- updated ``ExecutionContext`` to keep a reference to the ``runner``. This
  will enable Extenstions to do things like modify the job queue.
- Parameter now automatically convert int and boot kinds to integer and
  boolean respectively, this behavior can be supressed by specifying
  ``convert_types``=``False`` when defining the parameter.
- Fixed resource resolution when dependency location does not exist.
- All device ``push`` and ``pull`` commands now raise ``DeviceError`` if they
  didn't succeed.
- Fixed showing Parameter default of ``False`` for boolean values.
- Updated csv result processor with the option to use classifiers to
  add columns to ``results.csv``.
- ``wlauto.utils.formatter``: Fix terminal size discovery.
- The plugin loader will now follow symlinks.
- Added arm64-v8a to ABI map
- WA now reports syntax errors in a more informative way.
- Resource resolver: now prints the path of the found resource to the log.
- Resource getter: look for executable in the bin/ directory under resource
  owner's dependencies directory as well as general dependencies bin.
- ``GamingWorkload``:
     - Added an option to prevent clearing of package data before execution.
     - Added the ability to override the timeout of deploying the assets
       tarball.
- ``ApkWorkload``: Added an option to skip host-side APK check entirely.
- ``utils.misc.normalize``: only normalize string keys.
- Better error reporting for subprocess.CalledProcessError
- ``boolean`` now interprets ``'off'`` as ``False``
- ``wlauto.utils.uefi``: Added support for debug builds.
- ``wlauto.utils.serial_port``: Now supports fdexpect versions > 4.0.0
- Semanatics for ``initialize``/``finalize`` for *all* Plugins are changed
  so that now they will always run at most once per run. They will not be
  executed twice even if invoked via instances of different subclasses (if
  those subclasses defined their own verions, then their versions will be
  invoked once each, but the base version will only get invoked once).
- Pulling entries from procfs does not work on some platforms. WA now tries
  to cat the contents of a property_file and write it to a output file on the
  host.

Documentation
~~~~~~~~~~~~~
- ``installation``:
     - Added ``post install`` section which lists workloads that require
       additional external dependencies.
     - Added the ``uninstall`` and ``upgrade`` commands for users to remove or
       upgrade Workload Automation.
     - Added documentation explaining how to use ``remote_assets_path``
       setting.
     - Added warning about potential permission issues with pip.
- ``quickstart``: Added steps for setting up WA to run on Linux devices.
- ``device_setup``: fixed ``generic_linux`` ``device_config`` example.
- ``contributing``: Clarified style guidelines
- ``daq_device_setup``: Added an illustration for DAQ wiring.
- ``writing_plugins``: Documented the Workload initialize and finalize
  methods.
- Added descriptions to plugin that didn't have one.

Other
~~~~~
- ``daq_server``:
     - Fixed showing available devices.
     - Now works with earlier versions of the DAQmx driver.thus you can now run
       the server on Linux systems.
     - DAQ error messages are now properly propaged to the client.
     - Server will now periodically clean up uncollected files.
     - fixed not being able to resolve IP address for hostname
       (report "localhost" in that case).
     - Works with latest version of twisted.
- ``setup.py``: Fixed paths to work with Mac OS X.
- ``summary_csv`` is no longer enabled by default.
- ``status`` result processor is now enabled by default.
- Commands:
     - ``show``:
         - Now shows what platform plugins support.
         - Will no longer try to use a pager if ``PAGER=''`` in the environment.
     - ``list``:
         - Added ``"-p"`` option to filter results by supported platforms.
         - Added ``"--packaged-only"`` option to only list plugins packaged
           with WA.
     - ``run``: Added ``"--disable"`` option to diable instruments.
     - ``create``:
         - Added ``agenda`` sub-command to generate agendas for a set of
           plugins.
         - ``create workload`` now gives more informative errors if Android SDK
           installed but no platform has been downloaded.

Incompatible changes
####################

Framework
~~~~~~~~~
- ``BaseLinuxDevice``:
     - Renamed ``active_cpus`` to ``online_cpus``
     - Renamed ``get_cluster_cpu`` to ``get_cluster_active_cpu``
     - Renamed ``get_core_cpu`` to ``get_core_online_cpu``
- All plugin's ``initialize`` function now takes one (and only one)
  parameter, ``context``.
- ``wlauto.core.device``: Removed ``init`` function. Replaced with
  ``initialize``

-------------
Version 2.3.0
-------------

- First publicly-released version.
