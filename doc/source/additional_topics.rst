Additional Topics
+++++++++++++++++

Modules
=======

Modules are essentially plug-ins for Plugins. They provide a way of defining
common and reusable functionality. An Plugin can load zero or more modules
during its creation. Loaded modules will then add their capabilities (see
Capabilities_) to those of the Plugin. When calling code tries to access an
attribute of an Plugin the Plugin doesn't have, it will try to find the
attribute among its loaded modules and will return that instead.

.. note:: Modules are themselves plugins, and can therefore load their own
          modules. *Do not* abuse this.

For example, calling code may wish to reboot an unresponsive device by calling
``device.hard_reset()``, but the ``Device`` in question does not have a
``hard_reset`` method; however the ``Device`` has loaded ``netio_switch``
module which allows to disable power supply over a network (say this device
is in a rack and is powered through such a switch). The module has
``reset_power`` capability (see Capabilities_ below) and so implements
``hard_reset``. This will get invoked when ``device.hard_rest()`` is called.

.. note:: Modules can only extend Plugins with new attributes; they cannot
          override existing functionality. In the example above, if the
          ``Device`` has implemented ``hard_reset()`` itself, then *that* will
          get invoked irrespective of which modules it has loaded.

If two loaded modules have the same capability or implement the same method,
then the last module to be loaded "wins" and its method will be invoke,
effectively overriding the module that was loaded previously. 

Specifying Modules
------------------

Modules get loaded when an Plugin is instantiated by the plugin loader.
There are two ways to specify which modules should be loaded for a device.


Capabilities
============

Capabilities define the functionality that is implemented by an Plugin,
either within the Plugin itself or through loadable modules. A capability is
just a label, but there is an implied contract. When an Plugin claims to have
a particular capability, it promises to expose a particular set of
functionality through a predefined interface.

Currently used capabilities are described below.

.. note:: Since capabilities are basically random strings, the user can always
          define their own; and it is then up to the user to define, enforce and
          document the contract associated with their capability. Below, are the
          "standard" capabilities used in WA.


.. note:: The method signatures in the descriptions below show the calling
          signature (i.e. they're omitting the initial self parameter).

active_cooling
--------------

Intended to be used by devices and device modules, this capability implies 
that the device implements a controllable active cooling solution (e.g. 
a programmable fan). The device/module must implement the following methods: 

start_active_cooling()
        Active cooling is started (e.g. the fan is turned on)

stop_active_cooling()
        Active cooling is stopped (e.g. the fan is turned off)
        

reset_power
-----------

Intended to be used by devices and device modules, this capability implies 
that the device is capable of performing a hard reset by toggling power. The
device/module must implement the following method:

hard_reset()
        The device is restarted. This method cannot rely on the device being
        responsive and must work even if the software on the device has crashed.


flash
-----

Intended to be used by devices and device modules, this capability implies 
that the device can be flashed with new images.  The device/module must
implement the following method:

flash(image_bundle=None, images=None)
        ``image_bundle`` is a path to a "bundle" (e.g. a tarball) that contains
        all the images to be flashed. Which images go where must also be defined 
        within the bundle. ``images`` is a dict mapping image destination (e.g.
        partition name) to the path to that specific image. Both
        ``image_bundle`` and ``images`` may be specified at the same time. If
        there is overlap between the two, ``images`` wins and its contents will
        be flashed in preference to the ``image_bundle``.
