Modules
=======

Modules add additional functionality to the core :class:`Target` interface.
Usually, it is support for specific subsystems on the target. Modules are
instantiated as attributes of the :class:`Target` instance.

hotplug
-------

Kernel ``hotplug`` subsystem allows offlining ("removing") cores from the
system, and onlining them back int. The ``devlib`` module exposes a simple
interface to this subsystem

.. code:: python

   from devlib import LocalLinuxTarget
   target = LocalLinuxTarget()

   # offline cpus 2 and 3, "removing" them from the system
   target.hotplug.offline(2, 3)

   # bring CPU 2 back in
   target.hotplug.online(2)

   # Make sure all cpus are online
   target.hotplug.online_all()

cpufreq
-------

``cpufreq`` is the kernel subsystem for managing DVFS (Dynamic Voltage and
Frequency Scaling). It allows controlling frequency ranges and switching
policies (governors). The ``devlib`` module exposes the following interface

.. note:: On ARM big.LITTLE systems, all cores on a cluster (usually all cores
          of the same type) are in the same frequency domain, so setting
          ``cpufreq`` state on one core on a cluter will affect all cores on
          that cluster. Because of this, some devices only expose cpufreq sysfs
          interface (which is what is used by the ``devlib`` module) on the
          first cpu in a cluster. So to keep your scripts proable, always use
          the fist (online) CPU in a cluster to set ``cpufreq`` state.

.. method:: target.cpufreq.list_governors(cpu)

   List cpufreq governors available for the specified cpu. Returns a list of
   strings.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
               ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.list_governor_tunables(cpu)

   List the tunables for the specified cpu's current governor.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).


.. method:: target.cpufreq.get_governor(cpu)

   Returns the name of the currently set governor for the specified cpu.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
               ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.set_governor(cpu, governor, **kwargs)

   Sets the governor for the specified cpu.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
        ``1`` or ``"cpu1"``).
   :param governor: The name of the governor. This must be one of the governors 
                supported by the CPU (as retrunted by ``list_governors()``.

   Keyword arguments may be used to specify governor tunable values.


.. method:: target.cpufreq.get_governor_tunables(cpu)

   Return a dict with the values of the specfied CPU's current governor.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.set_governor_tunables(cpu, **kwargs)

   Set the tunables for the current governor on the specified CPU.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

   Keyword arguments should be used to specify tunable values.

.. method:: target.cpufreq.list_frequencie(cpu)

   List DVFS frequencies supported by the specified CPU. Returns a list of ints.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.get_min_frequency(cpu)
            target.cpufreq.get_max_frequency(cpu)
            target.cpufreq.set_min_frequency(cpu, frequency[, exact=True])
            target.cpufreq.set_max_frequency(cpu, frequency[, exact=True])

   Get and set min and max frequencies on the specfied CPU. "set" functions are
   avialable with all governors other than ``userspace``.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).
   :param frequency: Frequency to set.

.. method:: target.cpufreq.get_frequency(cpu)
            target.cpufreq.set_frequency(cpu, frequency[, exact=True])

   Get and set current frequency on the specified CPU. ``set_frequency`` is only
   available if the current governor is ``userspace``.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).
   :param frequency: Frequency to set.

cpuidle
-------

``cpufreq`` is the kernel subsystem for managing CPU low power (idle) states.

.. method:: taget.cpuidle.get_driver()

   Return the name current cpuidle driver.

.. method:: taget.cpuidle.get_governor()

   Return the name current cpuidle governor (policy).

.. method:: target.cpuidle.get_states([cpu=0])

   Return idle states (optionally, for the specified CPU). Returns a list of
   :class:`CpuidleState` instances.

.. method:: target.cpuidle.get_state(state[, cpu=0])

   Return :class:`CpuidleState` instance (optionally, for the specified CPU)
   representing the specified idle state. ``state`` can be either an integer
   index of the state or a string with the states ``name`` or ``desc``.

.. method:: target.cpuidle.enable(state[, cpu=0])
            target.cpuidle.disable(state[, cpu=0])
            target.cpuidle.enable_all([cpu=0])
            target.cpuidle.disable_all([cpu=0])

    Enable or disable the specified or all states (optionally on the specified
    CPU.

You can also call ``enable()`` or ``disable()`` on :class:`CpuidleState` objects 
returned by get_state(s).

cgroups
-------

TODO

hwmon
-----

TODO

API
---

TODO
