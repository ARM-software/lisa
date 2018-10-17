.. _writing-plugins:


Writing Plugins
================

Workload Automation offers several plugin points (or plugin types). The most
interesting of these are

:workloads: These are the tasks that get executed and measured on the device. These
            can be benchmarks, high-level use cases, or pretty much anything else.
:targets: These are interfaces to the physical devices (development boards or end-user
          devices, such as smartphones) that use cases run on. Typically each model of a
          physical device would require its own interface class (though some functionality
          may be reused by subclassing from an existing base).
:instruments: Instruments allow collecting additional data from workload execution (e.g.
              system traces). Instruments are not specific to a particular workload. Instruments
              can hook into any stage of workload execution.
:output processors: These are used to format the results of workload execution once they have been
                    collected. Depending on the callback used, these will run either after each
                    iteration and/or at the end of the run, after all of the results have been
                    collected.

You can create a plugin by subclassing the appropriate base class, defining
appropriate methods and attributes, and putting the .py file containing the
class into the "plugins" subdirectory under ``~/.workload_automation`` (or
equivalent) where it will be automatically picked up by WA.


Plugin Basics
--------------

This sub-section covers things common to implementing plugins of all types. It
is recommended you familiarize yourself with the information here before
proceeding onto guidance for specific plugin types.

.. _resource-resolution:

Dynamic Resource Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The idea is to decouple resource identification from resource discovery.
Workloads/instruments/devices/etc state *what* resources they need, and not
*where* to look for them -- this instead is left to the resource resolver that
is part of the execution context. The actual discovery of resources is
performed by resource getters that are registered with the resolver.

A resource type is defined by a subclass of
:class:`wa.framework.resource.Resource`. An instance of this class describes a
resource that is to be obtained. At minimum, a ``Resource`` instance has an
owner (which is typically the object that is looking for the resource), but
specific resource types may define other parameters that describe an instance of
that resource (such as file names, URLs, etc).

An object looking for a resource invokes a resource resolver with an instance of
``Resource`` describing the resource it is after. The resolver goes through the
getters registered for that resource type in priority order attempting to obtain
the resource; once the resource is obtained, it is returned to the calling
object. If none of the registered getters could find the resource,
``NotFoundError`` is raised (or ``None`` is returned instead, if invoked with
``strict=False``).

The most common kind of object looking for resources is a ``Workload``, and the
``Workload`` class defines
:py:meth:`wa.framework.workload.Workload.init_resources` method, which may be
overridden by subclasses to perform resource resolution. For example, a workload
looking for an executable file would do so like this::

    from wa import Workload
    from wa.import Executable

    class MyBenchmark(Workload):

        # ...

        def init_resources(self, resolver):
            resource = Executable(self, self.target.abi, 'my_benchmark')
            host_exe = resolver.get(resource)

        # ...


Currently available resource types are defined in :py:mod:`wa.framework.resources`.

.. _deploying-executables:

Deploying executables to a target
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some targets may have certain restrictions on where executable binaries may be
placed and how they should be invoked. To ensure your plugin works with as
wide a range of targets as possible, you should use WA APIs for deploying and
invoking executables on a target, as outlined below.

As with other resources, host-side paths to the executable binary to be deployed
should be obtained via the :ref:`resource resolver <resource-resolution>`. A
special resource type, ``Executable`` is used to identify  a binary to be
deployed. This is similar to the regular ``File`` resource, however it takes an
additional parameter that specifies the ABI for which the executable was
compiled for.

In order for the binary to be obtained in this way, it must be stored in one of
the locations scanned by the resource resolver in a directory structure
``<root>/bin/<abi>/<binary>`` (where ``root`` is the base resource location to
be searched, e.g. ``~/.workload_automation/dependencies/<plugin name>``, and
``<abi>`` is the ABI for which the executable has been compiled, as returned by
``self.target.abi``).

Once the path to the host-side binary has been obtained, it may be deployed
using one of two methods from a
`Target <http://devlib.readthedocs.io/en/latest/target.html>`_ instance --
``install`` or ``install_if_needed``. The latter will check a version of that
binary has been previously deployed by WA and will not try to re-install.

.. code:: python

  from wa import Executable

  host_binary = context.get(Executable(self, self.target.abi, 'some_binary'))
  target_binary = self.target.install_if_needed(host_binary)


.. note:: Please also note that the check is done based solely on the binary name.
          For more information please see the devlib
          `documentation <http://devlib.readthedocs.io/en/latest/target.html#Target.install_if_needed>`_.

Both of the above methods will return the path to the installed binary on the
target. The executable should be invoked *only* via that path; do **not** assume
that it will be in ``PATH`` on the target (or that the executable with the same
name in ``PATH`` is the version deployed by WA.

For more information on how to implement this, please see the
:ref:`how to guide <deploying-executables-example>`.


Deploying assets
-----------------
WA provides a generic mechanism for deploying assets during workload initialization.
WA will automatically try to retrieve and deploy each asset to the target's working directory
that is contained in a workloads ``deployable_assets`` attribute stored as a list.

If the parameter ``cleanup_assets`` is set then any asset deployed will be removed
again and the end of the run.

If the workload requires a custom deployment mechanism the ``deploy_assets``
method can be overridden for that particular workload, in which case, either
additional assets should have their on target paths added to the workload's
``deployed_assests`` attribute or the corresponding ``remove_assets`` method
should also be implemented.

.. _instrument-reference:

Adding an Instrument
---------------------
Instruments can be used to collect additional measurements during workload
execution (e.g. collect power readings). An instrument can hook into almost any
stage of workload execution. Any new instrument should be a subclass of
Instrument and it must have a name. When a new instrument is added to Workload
Automation, the methods of the new instrument will be found automatically and
hooked up to the supported signals. Once a signal is broadcasted, the
corresponding registered method is invoked.

Each method in ``Instrument`` must take two arguments, which are ``self`` and
``context``. Supported methods and their corresponding signals can be found in
the :ref:`Signals Documentation <instruments_method_map>`. To make
implementations easier and common, the basic steps to add new instrument is
similar to the steps to add new workload and an example can be found in the
:ref:`How To <adding-an-instrument-example>` section.

.. _instrument-api:

To implement your own instrument the relevant methods of the interface shown
below should be implemented:

    :name:

            The name of the instrument, this must be unique to WA.

    :description:

            A description of what the instrument can be used for.

    :parameters:

            A list of additional :class:`Parameters` the instrument can take.

    :initialize(context):

                This method will only be called once during the workload run
                therefore operations that only need to be performed initially should
                be performed here for example pushing the files to the target device,
                installing them.

    :setup(context):

                This method is invoked after the workload is setup. All the
                necessary setup should go inside this method. Setup, includes
                operations like clearing logs, additional configuration etc.

    :start(context):

                It is invoked just before the workload start execution. Here is
                where instrument measurement start being registered/taken.

    :stop(context):

                It is invoked just after the workload execution stops and where
                the measurements should stop being taken/registered.

    :update_output(context):

                This method is invoked after the workload updated its result and
                where the taken measures should be added to the result so it can be
                processed by WA.

    :teardown(context):

                It is invoked after the workload is torn down. It is a good place
                to clean any logs generated by the instrument.

    :finalize(context):

                This method is the complement to the initialize method and will also
                only be called once so should be used to deleting/uninstalling files
                pushed to the device.


This is similar to a ``Workload``, except all methods are optional. In addition to
the workload-like methods, instruments can define a number of other methods that
will get invoked at various points during run execution. The most useful of
which is perhaps ``initialize`` that gets invoked after the device has been
initialised for the first time, and can be used to perform one-time setup (e.g.
copying files to the device -- there is no point in doing that for each
iteration). The full list of available methods can be found in
:ref:`Signals Documentation <instruments_method_map>`.

.. _prioritization:

Prioritization
~~~~~~~~~~~~~~

Callbacks (e.g. ``setup()`` methods) for all instruments get executed at the
same point during workload execution, one after another. The order in which the
callbacks get invoked should be considered arbitrary and should not be relied
on (e.g. you cannot expect that just because instrument A is listed before
instrument B in the config, instrument A's callbacks will run first).

In some cases (e.g. in ``start()`` and ``stop()`` methods), it is important to
ensure that a particular instrument's callbacks run a closely as possible to the
workload's invocations in order to maintain accuracy of readings; or,
conversely, that a callback is executed after the others, because it takes a
long time and may throw off the accuracy of other instruments. You can do
this by using decorators on the appropriate methods. The available decorators are:
``very_slow``, ``slow``, ``normal``, ``fast``, ``very_fast``, with ``very_fast``
running closest to the workload invocation and ``very_slow`` running furtherest
away. For example::

    from wa import very_fast
    # ..

    class PreciseInstrument(Instrument)

        # ...
        @very_fast
        def start(self, context):
            pass

        @very_fast
        def stop(self, context):
            pass

        # ...

``PreciseInstrument`` will be started after all other instruments (i.e.
*just* before the workload runs), and it will stopped before all other
instruments (i.e. *just* after the workload runs).

If more than one active instrument has specified fast (or slow) callbacks, then
their execution order with respect to each other is not guaranteed. In general,
having a lot of instruments enabled is going to negatively affect the
readings. The best way to ensure accuracy of measurements is to minimize the
number of active instruments (perhaps doing several identical runs with
different instruments enabled).

Example
^^^^^^^

Below is a simple instrument that measures the execution time of a workload::

    class ExecutionTimeInstrument(Instrument):
        """
        Measure how long it took to execute the run() methods of a Workload.

        """

        name = 'execution_time'

        def initialize(self, context):
            self.start_time = None
            self.end_time = None

        @very_fast
        def start(self, context):
            self.start_time = time.time()

        @very_fast
        def stop(self, context):
            self.end_time = time.time()

        def update_output(self, context):
            execution_time = self.end_time - self.start_time
            context.add_metric('execution_time', execution_time, 'seconds')


.. include:: developer_information/developer_guide/instrument_method_map.rst

.. _adding-an-output-processor:

Adding an Output processor
----------------------------

A output processor is responsible for processing the results. This may
involve formatting and writing them to a file, uploading them to a database,
generating plots, etc. WA comes with a few output processors that output
results in a few common formats (such as csv or JSON).

You can add your own output processors by creating a Python file in
``~/.workload_automation/plugins`` with a class that derives from
:class:`wa.OutputProcessor <wa.framework.processor.OutputProcessor>`, and should
implement the relevant methods shown below, for more information and please
see the
:ref:`Adding an Output Processor <adding-an-output-processor-example>` section.

    :name:

            The name of the output processor, this must be unique to WA.

    :description:

            A description of what the output processor can be used for.

    :parameters:

            A list of additional :class:`Parameters` the output processor can take.

    :initialize(context):

                This method will only be called once during the workload run
                therefore operations that only need to be performed initially should
                be performed here.

    :process_job_output(output, target_info, run_ouput):

                This method should be used to perform the processing of the
                output from an individual job output. This is where any
                additional artifacts should be generated if applicable.

    :export_job_output(output, target_info, run_ouput):

                This method should be used to perform the exportation of the
                existing data collected/generated for an individual job. E.g.
                uploading them to a database etc.

    :process_run_output(output, target_info):

                This method should be used to perform the processing of the
                output from the run as a whole. This is where any
                additional artifacts should be generated if applicable.

    :export_run_output(output, target_info):

                This method should be used to perform the exportation of the
                existing data collected/generated for the run as a whole. E.g.
                uploading them to a database etc.

    :finalize(context):

                This method is the complement to the initialize method and will also
                only be called once.


The method names should be fairly self-explanatory. The difference between
"process" and "export" methods is that export methods will be invoked after
process methods for all output processors have been generated. Process methods
may generate additional artifacts (metrics, files, etc.), while export methods
should not -- they should only handle existing results (upload them to  a
database, archive on a filer, etc).

The output object passed to job methods is an instance of
:class:`wa.framework.output.JobOutput`, the output object passed to run methods
is an instance of :class:`wa.RunOutput <wa.framework.output.RunOutput>`.


Adding a Resource Getter
------------------------

A resource getter is a plugin that is designed to retrieve a resource
(binaries, APK files or additional workload assets). Resource getters are invoked in
priority order until one returns the desired resource.

If you want WA to look for resources somewhere it doesn't by default (e.g. you
have a repository of APK files), you can implement a getter for the resource and
register it with a higher priority than the standard WA getters, so that it gets
invoked first.

Instances of a resource getter should implement the following interface::

    class ResourceGetter(Plugin):

        name = None

        def register(self, resolver):
            raise NotImplementedError()

The getter should define a name for itself (as with all plugins), in addition it
should implement the ``register`` method. This involves registering a method
with the resolver that should used to be called when trying to retrieve a resource
(typically ``get``) along with it's priority (see `Getter Prioritization`_
below. That method should return an instance of the resource that
has been discovered (what "instance" means depends on the resource, e.g. it
could be a file path), or ``None`` if this getter was unable to discover
that resource.

Getter Prioritization
~~~~~~~~~~~~~~~~~~~~~

A priority is an integer with higher numeric values indicating a higher
priority. The following standard priority aliases are defined for getters:


    :preferred: Take this resource in favour of the environment resource.
    :local: Found somewhere under ~/.workload_automation/ or equivalent, or
            from environment variables, external configuration files, etc.
            These will override resource supplied with the package.
    :lan: Resource will be retrieved from a locally mounted remote location
          (such as samba share)
    :remote: Resource will be downloaded from a remote location (such as an HTTP
             server)
    :package: Resource provided with the package.

These priorities are defined as class members of
:class:`wa.framework.resource.SourcePriority`, e.g. ``SourcePriority.preferred``.

Most getters in WA will be registered with either ``local`` or
``package`` priorities. So if you want your getter to override the default, it
should typically be registered as ``preferred``.

You don't have to stick to standard priority levels (though you should, unless
there is a good reason). Any integer is a valid priority. The standard priorities
range from 0 to 40 in increments of 10.

Example
~~~~~~~

The following is an implementation of a getter that searches for files in the
users dependencies directory, typically
``~/.workload_automation/dependencies/<workload_name>`` It uses the
``get_from_location`` method to filter the available files in the provided
directory appropriately::

    import sys

    from wa import settings,
    from wa.framework.resource import ResourceGetter, SourcePriority
    from wa.framework.getters import get_from_location
    from wa.utils.misc import ensure_directory_exists as _d

    class UserDirectory(ResourceGetter):

        name = 'user'

        def register(self, resolver):
            resolver.register(self.get, SourcePriority.local)

        def get(self, resource):
            basepath = settings.dependencies_directory
            directory = _d(os.path.join(basepath, resource.owner.name))
            return get_from_location(directory, resource)

.. _adding_a_target:

Adding a Target
---------------

In WA3, a 'target' consists of a platform and a devlib target. The
implementations of the targets are located in ``devlib``. WA3 will instantiate a
devlib target passing relevant parameters parsed from the configuration. For
more information about devlib targets please see `the documentation
<http://devlib.readthedocs.io/en/latest/target.html>`_.

The currently available platforms are:
    :generic: The 'standard' platform implementation of the target, this should
              work for the majority of use cases.
    :juno: A platform implementation specifically for the juno.
    :tc2: A platform implementation specifically for the tc2.
    :gem5: A platform implementation to interact with a gem5 simulation.

The currently available targets from devlib are:
    :linux: A device running a Linux based OS.
    :android: A device running Android OS.
    :local: Used to run locally on a linux based host.
    :chromeos: A device running ChromeOS, supporting an android container if available.

For an example of adding you own customized version of an existing devlib target,
please see the how to section :ref:`Adding a Custom Target <adding-custom-target-example>`.


Other Plugin Types
---------------------

In addition to plugin types covered above, there are few other, more
specialized ones. They will not be covered in as much detail. Most of them
expose relatively simple interfaces with only a couple of methods and it is
expected that if the need arises to extend them, the API-level documentation
that accompanies them, in addition to what has been outlined here, should
provide enough guidance.

:commands: This allows extending WA with additional sub-commands (to supplement
           exiting ones outlined in the :ref:`invocation` section).
:modules: Modules are "plugins for plugins". They can be loaded by other
          plugins to expand their functionality (for example, a flashing
          module maybe loaded by a device in order to support flashing).


Packaging Your Plugins
----------------------

If your have written a bunch of plugins, and you want to make it easy to
deploy them to new systems and/or to update them on existing systems, you can
wrap them in a Python package. You can use ``wa create package`` command to
generate appropriate boiler plate. This will create a ``setup.py`` and a
directory for your package that you can place your plugins into.

For example, if you have a workload inside ``my_workload.py`` and an output
processor in ``my_output_processor.py``, and you want to package them as
``my_wa_exts`` package, first run the create command ::

        wa create package my_wa_exts

This will create a ``my_wa_exts`` directory which contains a
``my_wa_exts/setup.py`` and a subdirectory ``my_wa_exts/my_wa_exts`` which is
the package directory for your plugins (you can rename the top-level
``my_wa_exts`` directory to anything you like -- it's just a "container" for the
setup.py and the package directory). Once you have that, you can then copy your
plugins into the package directory, creating
``my_wa_exts/my_wa_exts/my_workload.py`` and
``my_wa_exts/my_wa_exts/my_output_processor.py``. If you have a lot of
plugins, you might want to organize them into subpackages, but only the
top-level package directory is created by default, and it is OK to have
everything in there.

.. note:: When discovering plugins through this mechanism, WA traverses the
          Python module/submodule tree, not the directory structure, therefore,
          if you are going to create subdirectories under the top level directory
          created for you, it is important that your make sure they are valid
          Python packages; i.e.  each subdirectory must contain a __init__.py
          (even if blank) in order for the code in that directory and its
          subdirectories to be discoverable.

At this stage, you may want to edit ``params`` structure near the bottom of
the ``setup.py`` to add correct author, license and contact information (see
"Writing the Setup Script" section in standard Python documentation for
details). You may also want to add a README and/or a COPYING file at the same
level as the setup.py.  Once you have the contents of your package sorted,
you can generate the package by running ::

        cd my_wa_exts
        python setup.py sdist

This  will generate ``my_wa_exts/dist/my_wa_exts-0.0.1.tar.gz`` package which
can then be deployed on the target system with standard Python package
management tools, e.g. ::

        sudo pip install my_wa_exts-0.0.1.tar.gz

As part of the installation process, the setup.py in the package, will write the
package's name into ``~/.workoad_automation/packages``. This will tell WA that
the package contains plugin and it will load them next time it runs.

.. note:: There are no uninstall hooks in ``setuputils``,  so if you ever
          uninstall your WA plugins package, you will have to manually remove
          it from ``~/.workload_automation/packages`` otherwise WA will complain
          about a missing package next time you try to run it.
