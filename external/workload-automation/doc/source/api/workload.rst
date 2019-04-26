.. _workloads-api:

Workloads
~~~~~~~~~
.. _workload-api:

Workload
^^^^^^^^

The base :class:`Workload` interface is as follows, and is the base class for
all :ref:`workload types <workload-types>`. For more information about to
implement your own workload please see the
:ref:`Developer How Tos <adding-a-workload-example>`.

All instances of a workload will have the following attributes:

``name``
   This identifies the workload (e.g. it is used to specify the
   workload in the :ref:`agenda <agenda>`).

``phones_home``
    This can be set to True to mark that this workload poses a risk of
    exposing information to the outside world about the device it runs on.
    For example a benchmark application that sends scores and device data
    to a database owned by the maintainer.

``requires_network``
    Set this to ``True`` to mark the the workload will fail without a network
    connection, this enables it to fail early with a clear message.

``asset_directory``
    Set this to specify a custom directory for assets to be pushed to, if
    unset the working directory will be used.

``asset_files``
    This can be used to automatically deploy additional assets to
    the device. If required the attribute should contain a list of file
    names that are required by the workload which will be attempted to be
    found by the resource getters

methods
"""""""

.. method:: Workload.init_resources(context)

    This method may be optionally overridden to implement dynamic
    resource discovery for the workload. This method executes
    early on, before the device has been initialized, so it
    should only be used to initialize resources that do not
    depend on the device to resolve. This method is executed
    once per run for each workload instance.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.validate(context)

    This method can be used to validate any assumptions your workload
    makes about the environment (e.g. that required files are
    present, environment variables are set, etc) and should raise a
    :class:`wa.WorkloadError <wa.framework.exception.WorkloadError>`
    if that is not the case. The base class implementation only makes
    sure sure that the name attribute has been set.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.initialize(context)

    This method is decorated with the ``@once_per_instance`` decorator,
    (for more information please see
    :ref:`Execution Decorators <execution-decorators>`)
    therefore it will be executed exactly once per run (no matter
    how many instances of the workload there are). It will run
    after the device has been initialized, so it may be used to
    perform device-dependent initialization that does not need to
    be repeated on each iteration (e.g. as installing executables
    required by the workload on the device).

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.setup(context)

    Everything that needs to be in place for workload execution should
    be done in this method. This includes copying files to the device,
    starting up an application, configuring communications channels,
    etc.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.setup_rerun(context)

    Everything that needs to be in place for workload execution should
    be done in this method. This includes copying files to the device,
    starting up an application, configuring communications channels,
    etc.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.run(context)

    This method should perform the actual task that is being measured.
    When this method exits, the task is assumed to be complete.

    :param context: The :ref:`Context <context>` for the current run.

    .. note:: Instruments are kicked off just before calling this
            method and disabled right after, so everything in this
            method is being measured. Therefore this method should
            contain the least code possible to perform the operations
            you are interested in measuring. Specifically, things like
            installing or starting applications, processing results, or
            copying files to/from the device should be done elsewhere if
            possible.



.. method:: Workload.extract_results(context)

    This method gets invoked after the task execution has finished and
    should be used to extract metrics from the target.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.update_output(context)

    This method should be used to update the output within the specified
    execution context with the metrics and artifacts from this
    workload iteration.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.teardown(context)

    This could be used to perform any cleanup you may wish to do, e.g.
    Uninstalling applications, deleting file on the device, etc.

    :param context: The :ref:`Context <context>` for the current run.


.. method:: Workload.finalize(context)

    This is the complement to ``initialize``. This will be executed
    exactly once at the end of the run. This should be used to
    perform any final clean up (e.g. uninstalling binaries installed
    in the ``initialize``)

    :param context: The :ref:`Context <context>` for the current run.

.. _apkworkload-api:

ApkWorkload
^^^^^^^^^^^^

The :class:`ApkWorkload` derives from the base :class:`Workload` class however
this associates the workload with a package allowing for an apk to be found for
the workload, setup and ran on the device before running the workload.

In addition to the attributes mentioned above ApkWorloads this class also
features the following attributes however this class does not present any new
methods.


``loading_time``
    This is the time in seconds that WA will wait for the application to load
    before continuing with the run. By default this will wait 10 second however
    if your application under test requires additional time this values should
    be increased.

``package_names``
    This attribute should be a list of Apk packages names that are
    suitable for this workload. Both the host (in the relevant resource
    locations) and device will be searched for an application with a matching
    package name.

``supported_versions``
    This attribute should be a list of apk versions that are suitable for this
    workload, if a specific apk version is not specified then any available
    supported version may be chosen.

``activity``
    This attribute can be optionally set to override the default activity that
    will be extracted from the selected APK file which will be used when
    launching the APK.

``view``
    This is the "view" associated with the application. This is used by
    instruments like ``fps`` to monitor the current framerate being generated by
    the application.

``apk``
    The is a :class:`PackageHandler`` which is what is used to store
    information about the apk and  manage the application itself, the handler is
    used to call the associated methods to manipulate the application itself for
    example to launch/close it etc.

``package``
    This is a more convenient way to access the package name of the Apk
    that was found and being used for the run.


.. _apkuiautoworkload-api:

ApkUiautoWorkload
^^^^^^^^^^^^^^^^^

The :class:`ApkUiautoWorkload` derives from :class:`ApkUIWorkload` which is an
intermediate class which in turn inherits from
:class:`ApkWorkload`, however in addition to associating an apk with the
workload this class allows for automating the application with UiAutomator.

This class define these additional attributes:

``gui``
    This attribute will be an instance of a :class:`UiAutmatorGUI` which is
    used to control the automation, and is what is used to pass parameters to the
    java class for example ``gui.uiauto_params``.


.. _apkreventworkload-api:

ApkReventWorkload
^^^^^^^^^^^^^^^^^

The :class:`ApkReventWorkload` derives from :class:`ApkUIWorkload` which is an
intermediate class which in turn inherits from
:class:`ApkWorkload`, however in addition to associating an apk with the
workload this class allows for automating the application with
:ref:`Revent <revent_files_creation>`.

This class define these additional attributes:

``gui``
    This attribute will be an instance of a :class:`ReventGUI` which is
    used to control the automation

``setup_timeout``
    This is the time allowed for replaying a recording for the setup stage.

``run_timeout``
    This is the time allowed for replaying a recording for the run stage.

``extract_results_timeout``
    This is the time allowed for replaying a recording for the extract results stage.

``teardown_timeout``
    This is the time allowed for replaying a recording for the teardown stage.


.. _uiautoworkload-api:

UiautoWorkload
^^^^^^^^^^^^^^

The :class:`UiautoWorkload` derives from :class:`UIWorkload` which is an
intermediate class which in turn inherits from
:class:`Workload`, however this allows for providing generic automation using
UiAutomator without associating a particular application with the workload.

This class define these additional attributes:

``gui``
    This attribute will be an instance of a :class:`UiAutmatorGUI` which is
    used to control the automation, and is what is used to pass parameters to the
    java class for example ``gui.uiauto_params``.


.. _reventworkload-api:

ReventWorkload
^^^^^^^^^^^^^^

The :class:`ReventWorkload` derives from :class:`UIWorkload` which is an
intermediate class which in turn inherits from
:class:`Workload`, however this allows for providing generic automation
using :ref:`Revent <revent_files_creation>` without associating with the
workload.

This class define these additional attributes:

``gui``
    This attribute will be an instance of a :class:`ReventGUI` which is
    used to control the automation

``setup_timeout``
    This is the time allowed for replaying a recording for the setup stage.

``run_timeout``
    This is the time allowed for replaying a recording for the run stage.

``extract_results_timeout``
    This is the time allowed for replaying a recording for the extract results stage.

``teardown_timeout``
    This is the time allowed for replaying a recording for the teardown stage.


