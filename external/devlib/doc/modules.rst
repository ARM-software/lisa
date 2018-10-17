.. _modules:

Modules
=======

Modules add additional functionality to the core :class:`Target` interface.
Usually, it is support for specific subsystems on the target. Modules are
instantiated as attributes of the :class:`Target` instance.

hotplug
-------

Kernel ``hotplug`` subsystem allows offlining ("removing") cores from the
system, and onlining them back in. The ``devlib`` module exposes a simple
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
          ``cpufreq`` state on one core on a cluster will affect all cores on
          that cluster. Because of this, some devices only expose cpufreq sysfs
          interface (which is what is used by the ``devlib`` module) on the
          first cpu in a cluster. So to keep your scripts portable, always use
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

.. method:: target.cpufreq.set_governor(cpu, governor, \*\*kwargs)

   Sets the governor for the specified cpu.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
        ``1`` or ``"cpu1"``).
   :param governor: The name of the governor. This must be one of the governors
                supported by the CPU (as returned by ``list_governors()``.

   Keyword arguments may be used to specify governor tunable values.


.. method:: target.cpufreq.get_governor_tunables(cpu)

   Return a dict with the values of the specified CPU's current governor.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.set_governor_tunables(cpu, \*\*kwargs)

   Set the tunables for the current governor on the specified CPU.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

   Keyword arguments should be used to specify tunable values.

.. method:: target.cpufreq.list_frequencies(cpu)

   List DVFS frequencies supported by the specified CPU. Returns a list of ints.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.get_min_frequency(cpu)
            target.cpufreq.get_max_frequency(cpu)
            target.cpufreq.set_min_frequency(cpu, frequency[, exact=True])
            target.cpufreq.set_max_frequency(cpu, frequency[, exact=True])

   Get the currently set, or set new min and max frequencies for the specified
   CPU. "set" functions are available with all governors other than
   ``userspace``.

   :param cpu: The cpu; could be a numeric or the corresponding string (e.g.
       ``1`` or ``"cpu1"``).

.. method:: target.cpufreq.get_min_available_frequency(cpu)
            target.cpufreq.get_max_available_frequency(cpu)

    Retrieve the min or max DVFS frequency that is supported (as opposed to
    currently enforced) for a given CPU. Returns an int or None if could not be
    determined.

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

``cpuidle`` is the kernel subsystem for managing CPU low power (idle) states.

.. method:: target.cpuidle.get_driver()

   Return the name current cpuidle driver.

.. method:: target.cpuidle.get_governor()

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

Generic Module API Description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modules implement discrete, optional pieces of functionality ("optional" in the
sense that the functionality may or may not be present on the target device, or
that it may or may not be necessary for a particular application).

Every module (ultimately) derives from :class:`Module` class.  A module must
define the following class attributes:

:name: A unique name for the module. This cannot clash with any of the existing
       names and must be a valid Python identifier, but is otherwise free-form.
:kind: This identifies the type of functionality a module implements, which in
       turn determines the interface implemented by the module (all modules of
       the same kind must expose a consistent interface). This must be a valid
       Python identifier, but is otherwise free-form, though, where possible,
       one should try to stick to an already-defined kind/interface, lest we end
       up with a bunch of modules implementing similar functionality but
       exposing slightly different interfaces.

       .. note:: It is possible to omit ``kind`` when defining a module, in
                 which case the module's ``name`` will be treated as its
                 ``kind`` as well.

:stage: This defines when the module will be installed into a :class:`Target`.
        Currently, the following values are allowed:

        :connected: The module is installed after a connection to the target has
                    been established. This is the default.
        :early: The module will be installed when a :class:`Target` is first
                created. This should be used for modules that do not rely on a
                live connection to the target.
        :setup: The module will be installed after initial setup of the device
                has been performed. This allows the module to utilize assets
                deployed during the setup stage for example 'Busybox'.

Additionally, a module must implement a static (or class) method :func:`probe`:

.. method:: Module.probe(target)

    This method takes a :class:`Target` instance and returns ``True`` if this
    module is supported by that target, or ``False`` otherwise.

    .. note:: If the module ``stage`` is ``"early"``, this method cannot assume
              that a connection has been established (i.e. it can only access
              attributes of the Target that do not rely on a connection).

Installation and invocation
***************************

The default installation method will create an instance of a module (the
:class:`Target` instance being the sole argument) and assign it to the target
instance attribute named after the module's ``kind`` (or ``name`` if ``kind`` is
``None``).

It is possible to change the installation procedure for a module by overriding
the default :func:`install` method. The method must have the following
signature:

.. method:: Module.install(cls, target, **kwargs)

    Install the module into the target instance.


Implementation and Usage Patterns
*********************************

There are two common ways to implement the above API, corresponding to the two
common uses for modules:

- If a module provides an interface to a particular set of functionality (e.g.
  an OS subsystem), that  module would typically derive directly form
  :class:`Module` and  would leave ``kind`` unassigned, so that it is accessed
  by it name. Its instance's methods and attributes provide the interface for
  interacting with its functionality. For examples of this type of module, see
  the subsystem modules listed above (e.g. ``cpufreq``).
- If a module provides a platform- or infrastructure-specific implementation of
  a common function, the module would derive from one of :class:`Module`
  subclasses that define the interface for that function. In that case the
  module would be accessible via the common ``kind`` defined its super. The
  module would typically implement :func:`__call__` and be invoked directly. For
  examples of this type of module, see common function interface definitions
  below.


Common Function Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

This section documents :class:`Module` classes defining interface for common
functions. Classes derived from them provide concrete implementations for
specific platforms.


HardResetModule
***************

.. attribute:: HardResetModule.kind

    "hard_reset"

.. method:: HardResetModule.__call__()

    Must be implemented by derived classes.

    Implements hard reset for a target devices. The equivalent of physically
    power cycling the device.  This may be used by client code in situations
    where the target becomes unresponsive and/or a regular reboot is not
    possible.


BootModule
**********

.. attribute:: BootModule.kind

    "hard_reset"

.. method:: BootModule.__call__()

    Must be implemented by derived classes.

    Implements a boot procedure. This takes the device from (hard or soft)
    reset to a booted state where the device is ready to accept connections. For
    a lot of commercial devices the process is entirely automatic, however some
    devices (e.g. development boards), my require additional steps, such as
    interactions with the bootloader, in order to boot into the OS.

.. method:: Bootmodule.update(\*\*kwargs)

    Update the boot settings. Some boot sequences allow specifying settings
    that will be utilized during boot (e.g. linux kernel boot command line). The
    default implementation will set each setting in ``kwargs`` as an attribute of
    the boot module (or update the existing attribute).


FlashModule
***********

.. attribute:: FlashModule.kind

    "flash"

.. method:: __call__(image_bundle=None, images=None, boot_config=None)

    Must be implemented by derived classes.

    Flash the target platform with the specified images.

    :param image_bundle: A compressed bundle of image files with any associated
                         metadata. The format of the bundle is specific to a
                         particular implementation.
    :param images: A dict mapping image names/identifiers to the path on the
                   host file system of the corresponding image file. If both
                   this and ``image_bundle`` are specified, individual images
                   will override those in the bundle.
    :param boot_config: Some platforms require specifying boot arguments at the
                        time of flashing the images, rather than during each
                        reboot. For other platforms, this will be ignored.


Module Registration
~~~~~~~~~~~~~~~~~~~

Modules are specified on :class:`Target` or :class:`Platform` creation by name.
In order to find the class associated with the name, the module needs to be
registered with ``devlib``. This is accomplished by passing the module class
into :func:`register_module` method once it is defined.

.. note:: If you're wiring a module to be included as part of ``devlib`` code
          base, you can place the file with the module class under
          ``devlib/modules/`` in the source and it will be automatically
          enumerated. There is no need to explicitly register it in that case.

The code snippet below illustrates an implementation of a hard reset function
for an "Acme" device.

.. code:: python

    import os
    from devlib import HardResetModule, register_module


    class AcmeHardReset(HardResetModule):

        name = 'acme_hard_reset'

        def __call__(self):
            # Assuming Acme board comes with a "reset-acme-board" utility
            os.system('reset-acme-board {}'.format(self.target.name))

    register_module(AcmeHardReset)

