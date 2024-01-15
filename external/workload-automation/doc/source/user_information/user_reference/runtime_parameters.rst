.. _runtime-parameters:

Runtime Parameters
------------------

.. contents:: Contents
   :local:

Runtime parameters are options that can be specified to automatically configure
device at runtime. They can be specified at the global level in the agenda or
for individual workloads.

Example
^^^^^^^
Say we want to perform an experiment on an Android big.LITTLE devices to compare
the power consumption between the big and LITTLE clusters running the dhrystone
and benchmarkpi workloads. Assuming we have additional instrumentation active
for this device that can measure the power the device is consuming, to reduce
external factors we want to ensure that the device is in airplane mode turned on
for all our tests and the screen is off only for our dhrystone run. We will then
run 2 :ref:`sections <sections>` will each enable a single cluster on the
device, set the cores to their maximum frequency and disable all available idle
states.

.. code-block:: yaml

        config:
            runtime_parameters:
                  airplane_mode: true
        #..
        workloads:
                - name: dhrystone
                  iterations: 1
                  runtime_parameters:
                        screen_on: false
                        unlock_screen: 'vertical'
                - name: benchmarkpi
                  iterations: 1
        sections:
                - id: LITTLES
                  runtime_parameters:
                        num_little_cores: 4
                        little_governor: userspace
                        little_frequency: max
                        little_idle_states: none
                        num_big_cores: 0

                - id: BIGS
                  runtime_parameters:
                        num_big_cores: 4
                        big_governor: userspace
                        big_frequency: max
                        big_idle_states: none
                        num_little_cores: 0


HotPlug
^^^^^^^

Parameters:

:num_cores: An ``int`` that specifies the total number of cpu cores to be online.

:num_<core_name>_cores: An ``int`` that specifies the total number of that particular core
                              to be online, the target will be queried and if the core_names can
                              be determine a parameter for each of the unique core names will be
                              available.

:cpu<core_no>_online: A ``boolean`` that specifies whether that particular cpu, e.g. cpu0 will
                            be online.

If big.LITTLE is detected for the device and additional 2 parameters are available:

:num_big_cores: An ``int`` that specifies the total number of `big` cpu cores to be online.

:num_little_cores: An ``int`` that specifies the total number of `little` cpu cores to be online.



.. Note:: Please note that if the device in question is operating its own dynamic
          hotplugging then WA may be unable to set the CPU state or will be overridden.
          Unfortunately the method of disabling dynamic hot plugging will vary from
          device to device.


CPUFreq
^^^^^^^

:frequency: An ``int`` that can be used to specify a frequency for all cores if there are common frequencies available.

.. Note:: When settings the frequency, if the governor is not set to userspace then WA will attempt to set the maximum
          and minimum frequencies to mimic the desired behaviour.

:max_frequency: An ``int`` that can be used to specify a maximum frequency for all cores if there are common frequencies available.

:min_frequency: An ``int`` that can be used to specify a minimum frequency for all cores if there are common frequencies available.

:governor: A ``string`` that can be used to specify the governor for all cores if there are common governors available.

:governor: A ``string`` that can be used to specify the governor for all cores if there are common governors available.

:gov_tunables: A ``dict`` that can be used to specify governor
                   tunables for all cores, unlike the other common parameters these are not
                   validated at the beginning of the run therefore incorrect values will cause
                   an error during runtime.

:<core_name>_frequency: An ``int`` that can be used to specify a frequency for cores of a particular type e.g. 'A72'.

:<core_name>_max_frequency: An ``int`` that can be used to specify a maximum frequency for cores of a particular type e.g. 'A72'.

:<core_name>_min_frequency: An ``int`` that can be used to specify a minimum frequency for cores of a particular type e.g. 'A72'.

:<core_name>_governor: A ``string`` that can be used to specify the governor for cores of a particular type e.g. 'A72'.

:<core_name>_governor: A ``string`` that can be used to specify the governor for cores of a particular type e.g. 'A72'.

:<core_name>_gov_tunables: A ``dict`` that can be used to specify governor
                         tunables for cores of a particular type e.g. 'A72', these are not
                         validated at the beginning of the run therefore incorrect values will cause
                         an error during runtime.


:cpu<no>_frequency: An ``int`` that can be used to specify a frequency for a particular core e.g. 'cpu0'.

:cpu<no>_max_frequency: An ``int`` that can be used to specify a maximum frequency for a particular core e.g. 'cpu0'.

:cpu<no>_min_frequency: An ``int`` that can be used to specify a minimum frequency for a particular core e.g. 'cpu0'.

:cpu<no>_governor: A ``string`` that can be used to specify the governor for a particular core e.g. 'cpu0'.

:cpu<no>_governor: A ``string`` that can be used to specify the governor for a particular core e.g. 'cpu0'.

:cpu<no>_gov_tunables: A ``dict`` that can be used to specify governor
                         tunables for a particular core e.g. 'cpu0', these are not
                         validated at the beginning of the run therefore incorrect values will cause
                         an error during runtime.


If big.LITTLE is detected for the device an additional set of parameters are available:

:big_frequency: An ``int`` that can be used to specify a frequency for the big cores.

:big_max_frequency: An ``int`` that can be used to specify a maximum frequency for the big cores.

:big_min_frequency: An ``int`` that can be used to specify a minimum frequency for the big cores.

:big_governor: A ``string`` that can be used to specify the governor for the big cores.

:big_governor: A ``string`` that can be used to specify the governor for the big cores.

:big_gov_tunables: A ``dict`` that can be used to specify governor
                         tunables for the big cores, these are not
                         validated at the beginning of the run therefore incorrect values will cause
                         an error during runtime.

:little_frequency: An ``int`` that can be used to specify a frequency for the little cores.

:little_max_frequency: An ``int`` that can be used to specify a maximum frequency for the little cores.

:little_min_frequency: An ``int`` that can be used to specify a minimum frequency for the little cores.

:little_governor: A ``string`` that can be used to specify the governor for the little cores.

:little_governor: A ``string`` that can be used to specify the governor for the little cores.

:little_gov_tunables: A ``dict`` that can be used to specify governor
                         tunables for the little cores, these are not
                         validated at the beginning of the run therefore incorrect values will cause
                         an error during runtime.


CPUIdle
^^^^^^^

:idle_states: A ``string`` or list of strings which can be used to specify what
            idles states should be enabled for all cores if there are common
            idle states available. 'all' and 'none' are also valid entries as a
            shorthand

:<core_name>_idle_states: A ``string`` or list of strings which can be used to
                          specify what idles states should be enabled for cores of a particular type
                          e.g. 'A72'. 'all' and 'none' are also valid entries as a shorthand
:cpu<no>_idle_states: A ``string`` or list of strings which can be used to
                      specify what idles states should be enabled for a particular core e.g.
                      'cpu0'. 'all' and 'none' are also valid entries as a shorthand

If big.LITTLE is detected for the device and additional set of parameters are available:

:big_idle_states: A ``string`` or list of strings which can be used to specify
    what idles states should be enabled for the big cores. 'all' and 'none' are
    also valid entries as a shorthand
:little_idle_states: A ``string`` or list of strings which can be used to
    specify what idles states should be enabled for the little cores. 'all' and
    'none' are also valid entries as a shorthand.


Android Specific Runtime Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:brightness: An ``int`` between 0 and 255 (inclusive) to specify the brightness
    the screen should be set to. Defaults to ``127``.

:airplane_mode: A ``boolean`` to specify whether airplane mode should be
    enabled for the device.

:rotation: A ``String`` to specify the screen orientation for the device. Valid
    entries are ``NATURAL``, ``LEFT``, ``INVERTED``, ``RIGHT``.

:screen_on: A ``boolean`` to specify whether the devices screen should be
    turned on. Defaults to ``True``.

:unlock_screen: A ``String`` to specify how the devices screen should be
    unlocked. Unlocking screen is disabled by default. ``vertical``, ``diagonal``
    and ``horizontal`` are the supported values (see :meth:`devlib.AndroidTarget.swipe_to_unlock`).
    Note that unlocking succeeds when no passcode is set. Since unlocking screen
    requires turning on the screen, this option overrides value of ``screen_on``
    option.

.. _setting-sysfiles:

Setting Sysfiles
^^^^^^^^^^^^^^^^
In order to perform additional configuration of a target the ``sysfile_values``
runtime parameter can be used. The value for this parameter is a mapping (an
associative array, in YAML) of file paths onto values that should be written
into those files. ``sysfile_values`` is the only runtime parameter that is
available for any (Linux) device. Other runtime parameters will depend on the
specifics of the device used (e.g. its CPU cores configuration) as detailed
above.

.. note:: By default WA will attempt to verify that the any sysfile values were
   written correctly by reading the node back and comparing the two values. If
   you do not wish this check to happen, for example the node you are writing to
   is write only, you can append an ``!`` to the file path to disable this
   verification.

For example the following configuration could be used to enable and verify that cpu0
is online, however will not attempt to check that its governor have been set to
userspace::

                - name: dhrystone
                runtime_params:
                      sysfile_values:
                            /sys/devices/system/cpu/cpu0/online: 1
                            /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor!: userspace
