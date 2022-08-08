.. _output_processing_api:

Output
======

A WA output directory can be accessed via a :class:`RunOutput` object. There are
two ways of getting one -- either instantiate it with a path to a WA output
directory, or use :func:`discover_wa_outputs` to traverse a directory tree
iterating over all WA output directories found.

.. function:: discover_wa_outputs(path)

    Recursively traverse ``path`` looking for WA output directories. Return
    an iterator over :class:`RunOutput` objects for each discovered output.

    :param path: The directory to scan for WA output


.. class:: RunOutput(path)

    The main interface into a WA output directory.

    :param path: must be the path to the top-level output directory (the one
                 containing ``__meta`` subdirectory and ``run.log``).

WA output stored in a Postgres database by the ``Postgres`` output processor
can be accessed via a :class:`RunDatabaseOutput` which can be initialized as follows:

.. class:: RunDatabaseOutput(password, host='localhost', user='postgres', port='5432', dbname='wa', run_uuid=None, list_runs=False)

    The main interface into Postgres database containing WA results.

    :param password: The password used to authenticate with
    :param host: The database host address. Defaults to ``'localhost'``
    :param user: The user name used to authenticate with. Defaults to ``'postgres'``
    :param port: The database connection port number. Defaults to ``'5432'``
    :param dbname: The database name. Defaults to ``'wa'``
    :param run_uuid: The ``run_uuid`` to identify the selected run
    :param list_runs: Will connect to the database and will print out the available runs
            with their corresponding run_uuids. Defaults to ``False``


Example
-------

.. seealso:: :ref:`processing_output`

To demonstrate how we can use the output API if we have an existing WA output
called ``wa_output`` in the current working directory we can initialize a
``RunOutput`` as follows:

.. code-block:: python

    In [1]: from wa import RunOutput
       ...:
       ...: output_directory = 'wa_output'
       ...: run_output = RunOutput(output_directory)

Alternatively if the results have been stored in a Postgres database we can
initialize a ``RunDatabaseOutput`` as follows:

.. code-block:: python

    In [1]: from wa import RunDatabaseOutput
       ...:
       ...: db_settings = {
       ...:                host: 'localhost',
       ...:                port: '5432',
       ...:                dbname: 'wa'
       ...:                user: 'postgres',
       ...:                password: 'wa'
       ...:                }
       ...:
       ...: RunDatabaseOutput(list_runs=True, **db_settings)
    Available runs are:
    ========= ============ ============= =================== =================== ====================================
     Run Name      Project Project Stage          Start Time            End Time                             run_uuid
    ========= ============ ============= =================== =================== ====================================
    Test Run    my_project          None 2018-11-29 14:53:08 2018-11-29 14:53:24 aa3077eb-241a-41d3-9610-245fd4e552a9
    run_1       my_project          None 2018-11-29 14:53:34 2018-11-29 14:53:37 4c2885c9-2f4a-49a1-bbc5-b010f8d6b12a
    ========= ============ ============= =================== =================== ====================================

    In [2]: run_uuid = '4c2885c9-2f4a-49a1-bbc5-b010f8d6b12a'
       ...: run_output = RunDatabaseOutput(run_uuid=run_uuid, **db_settings)


From here we can retrieve various information about the run. For example if we
want to see what the overall status of the run was, along with the runtime
parameters and the metrics recorded from the first job was we can do the following:

.. code-block:: python

    In [2]: run_output.status
    Out[2]: OK(7)

    # List all of the jobs for the run
    In [3]: run_output.jobs
    Out[3]:
    [<wa.framework.output.JobOutput at 0x7f70358a1f10>,
     <wa.framework.output.JobOutput at 0x7f70358a1150>,
     <wa.framework.output.JobOutput at 0x7f7035862810>,
     <wa.framework.output.JobOutput at 0x7f7035875090>]

    # Examine the first job that was ran
    In [4]: job_1 = run_output.jobs[0]

    In [5]: job_1.label
    Out[5]: u'dhrystone'

    # Print out all the runtime parameters and their values for this job
    In [6]: for k, v in job_1.spec.runtime_parameters.items():
       ...:     print (k, v)
    (u'airplane_mode': False)
    (u'brightness': 100)
    (u'governor': 'userspace')
    (u'big_frequency': 1700000)
    (u'little_frequency': 1400000)

    # Print out all the metrics available for this job
    In [7]: job_1.metrics
    Out[7]:
    [<thread 0 score: 14423105 (+)>,
     <thread 0 DMIPS: 8209 (+)>,
     <thread 1 score: 14423105 (+)>,
     <thread 1 DMIPS: 8209 (+)>,
     <thread 2 score: 14423105 (+)>,
     <thread 2 DMIPS: 8209 (+)>,
     <thread 3 score: 18292638 (+)>,
     <thread 3 DMIPS: 10411 (+)>,
     <thread 4 score: 17045532 (+)>,
     <thread 4 DMIPS: 9701 (+)>,
     <thread 5 score: 14150917 (+)>,
     <thread 5 DMIPS: 8054 (+)>,
     <time: 0.184497 seconds (-)>,
     <total DMIPS: 52793 (+)>,
     <total score: 92758402 (+)>]

    # Load the run results csv file into pandas
    In [7]: pd.read_csv(run_output.get_artifact_path('run_result_csv'))
    Out[7]:
                id   workload  iteration          metric          value    units
    0   450000-wk1  dhrystone          1  thread 0 score   1.442310e+07      NaN
    1   450000-wk1  dhrystone          1  thread 0 DMIPS   8.209700e+04      NaN
    2   450000-wk1  dhrystone          1  thread 1 score   1.442310e+07      NaN
    3   450000-wk1  dhrystone          1  thread 1 DMIPS   8.720900e+04      NaN
    ...


We can also retrieve information about the target that the run was performed on
for example:

.. code-block:: python

    # Print out the target's abi:
    In [9]: run_output.target_info.abi
    Out[9]: u'arm64'

    # The os the target was running
    In [9]: run_output.target_info.os
    Out[9]: u'android'

    # And other information about the os version
    In [10]: run_output.target_info.os_version
    Out[10]:
    OrderedDict([(u'all_codenames', u'REL'),
                 (u'incremental', u'3687331'),
                 (u'preview_sdk', u'0'),
                 (u'base_os', u''),
                 (u'release', u'7.1.1'),
                 (u'codename', u'REL'),
                 (u'security_patch', u'2017-03-05'),
                 (u'sdk', u'25')])



:class:`RunOutput`
------------------

:class:`RunOutput` provides access to the output of a WA :term:`run`, including metrics,
artifacts, metadata, and configuration. It has the following attributes:


``jobs``
    A list of :class:`JobOutput` objects for each job that was executed during
    the run.

``status``
    Run status. This indicates whether the run has completed without problems
    (``Status.OK``) or if there were issues.

``metrics``
    A list of :class:`Metric`\ s for the run.

    .. note:: these are *overall run* metrics only. Metrics for individual
              jobs are contained within the corresponding :class:`JobOutput`\ s.

``artifacts``
    A list of :class:`Artifact`\ s for the run. These are usually backed by a
    file and can contain traces, raw data, logs, etc.

    .. note:: these are *overall run* artifacts only. Artifacts for individual
              jobs are contained within the corresponding :class:`JobOutput`\ s.

``info``
  A :ref:`RunInfo <run-info-api>` object that contains information about the run
  itself for example it's duration, name, uuid etc.

``target_info``
  A :ref:`TargetInfo <target-info-api>` object which can be used to access
  various information about the target that was used during the run for example
  it's ``abi``, ``hostname``, ``os`` etc.

``run_config``
  A :ref:`RunConfiguration <run-configuration>` object that can be used to
  access all the configuration of the run itself, for example the
  ``reboot_policy``, ``execution_order``, ``device_config`` etc.

``classifiers``
  :ref:`classifiers <classifiers>` defined for the entire run.

``metadata``
  :ref:`metadata  <metadata>` associated with the run.

``events``
  A list of any events logged during the run, that are not associated with a
  particular job.

``event_summary``
  A condensed summary of any events that occurred during the run.

``augmentations``
  A list of the :term:`augmentation`\ s that were enabled during the run (these
  augmentations may or may not have been active for a particular job).

``basepath``
  A (relative) path to the WA output directory backing this object.


methods
~~~~~~~

.. method:: RunOutput.get_artifact(name)

    Return the :class:`Artifact` specified by ``name``. This will only look
    at the run artifacts; this will not search the artifacts of the individual
    jobs.

    :param name:  The name of the artifact who's path to retrieve.
    :return: The :class:`Artifact` with that name
    :raises HostError: If the artifact with the specified name does not exist.


.. method:: RunOutput.get_artifact_path(name)

    Return the path to the file backing the artifact specified by ``name``. This
    will only look at the run artifacts; this will not search the artifacts of
    the individual jobs.

    :param name:  The name of the artifact who's path to retrieve.
    :return: The path to the artifact
    :raises HostError: If the artifact with the specified name does not exist.


.. method:: RunOutput.get_metric(name)

   Return the :class:`Metric` associated with the run (not the individual jobs)
   with the specified `name`.

   :return: The :class:`Metric` object for the metric with the specified name.


.. method:: RunOutput.get_job_spec(spec_id)

   Return the :class:`JobSpec` with the specified `spec_id`. A :term:`spec`
   describes the job to be executed. Each :class:`Job` has an associated
   :class:`JobSpec`, though a single :term:`spec` can be associated with
   multiple :term:`job`\ s (If the :term:`spec` specifies multiple iterations).

.. method:: RunOutput.list_workloads()

    List unique  workload labels that featured in this run. The labels will be
    in the order in which they first ran.

    :return: A list of `str` labels of workloads that were part of this run.


.. method:: RunOutput.add_classifier(name, value, overwrite=False)

   Add a classifier to the run as a whole. If a classifier with the specified
   ``name`` already exists, a``ValueError`` will be raised, unless
   `overwrite=True` is specified.


:class:`RunDatabaseOutput`
---------------------------

:class:`RunDatabaseOutput` provides access to the output of a WA :term:`run`,
including metrics,artifacts, metadata, and configuration stored in a postgres database.
The majority of attributes and methods are the same :class:`RunOutput` however the
noticeable differences are:

``jobs``
    A list of :class:`JobDatabaseOutput` objects for each job that was executed
    during the run.

``basepath``
  A representation of the current database and host information backing this object.

methods
~~~~~~~

.. method:: RunDatabaseOutput.get_artifact(name)

    Return the :class:`Artifact` specified by ``name``. This will only look
    at the run artifacts; this will not search the artifacts of the individual
    jobs. The `path` attribute of the :class:`Artifact` will be set to the Database OID of the object.

    :param name:  The name of the artifact who's path to retrieve.
    :return: The :class:`Artifact` with that name
    :raises HostError: If the artifact with the specified name does not exist.


.. method:: RunDatabaseOutput.get_artifact_path(name)

    If the artifcat is a file this method returns a `StringIO` object containing
    the contents of the artifact specified by ``name``. If the aritifcat is a
    directory, the method returns a path to a locally extracted version of the
    directory which is left to the user to remove after use. This will only look
    at the run artifacts; this will not search the artifacts of the individual
    jobs.

    :param name:  The name of the artifact who's path to retrieve.
    :return: A `StringIO` object with the contents of the artifact
    :raises HostError: If the artifact with the specified name does not exist.


:class:`JobOutput`
------------------

:class:`JobOutput` provides access to the output of a single :term:`job`
executed during a WA :term:`run`, including metrics,
artifacts, metadata, and configuration. It has the following attributes:

``status``
    Job status. This indicates whether the job has completed without problems
    (``Status.OK``) or if there were issues.

    .. note:: Under typical configuration, WA will make a number of attempts to
              re-run a job in case of issue. This status (and the rest of the
	      output) will represent the the latest attempt. I.e. a
	      ``Status.OK`` indicates that the latest attempt was successful,
	      but it does mean that there weren't prior failures. You can check
	      the ``retry`` attribute (see below) to whether this was the first
	      attempt or not.

``retry``
   Retry number for this job. If a problem is detected during job execution, the
   job will be re-run up to :confval:`max_retries` times. This indicates the
   final retry number for the output. A value of ``0`` indicates that the job
   succeeded on the first attempt, and no retries were necessary.

   .. note:: Outputs for previous attempts are moved into ``__failed``
             subdirectory of WA output. These are currently not exposed via the
	     API.

``id``
    The ID of the :term:`spec` associated with with job. This ID is unique to
    the spec, but not necessary to the job -- jobs representing multiple
    iterations of the same spec will share the ID.

``iteration``
    The iteration number of this job. Together with the ``id`` (above), this
    uniquely identifies a job with a run.

``label``
    The workload label associated with this job. Usually, this will be the name
    or :term:`alias` of the workload, however maybe overwritten by the user in
    the :term:`agenda`.

``metrics``
    A list of :class:`Metric`\ s for the job.

``artifacts``
    A list of :class:`Artifact`\ s for the job These are usually backed by a
    file and can contain traces, raw data, logs, etc.

``classifiers``
  :ref:`classifiers <classifiers>` defined for the job.

``metadata``
  :ref:`metadata  <metadata>` associated with the job.

``events``
  A list of any events logged during the execution of the job.

``event_summary``
  A condensed summary of any events that occurred during the execution of the
  job.

``augmentations``
  A list of the :term:`augmentation`\ s that were enabled for this job. This may
  be different from overall augmentations specified for the run, as they may be
  enabled/disabled on per-job basis.

``basepath``
  A (relative) path to the WA output directory backing this object.


methods
~~~~~~~

.. method:: JobOutput.get_artifact(name)

    Return the :class:`Artifact` specified by ``name`` associated with this job.

    :param name:  The name of the artifact to retrieve.
    :return: The :class:`Artifact` with that name
    :raises HostError: If the artifact with the specified name does not exist.

.. method:: JobOutput.get_artifact_path(name)

    Return the path to the file backing the artifact specified by ``name``,
    associated with this job.

    :param name:  The name of the artifact who's path to retrieve.
    :return: The path to the artifact
    :raises HostError: If the artifact with the specified name does not exist.

.. method:: JobOutput.get_metric(name)

   Return the :class:`Metric` associated with this job with the specified
   `name`.

   :return: The :class:`Metric` object for the metric with the specified name.

.. method:: JobOutput.add_classifier(name, value, overwrite=False)

   Add a classifier to the job. The classifier will be propagated to all
   existing artifacts and metrics, as well as those added afterwards. If a
   classifier with the specified ``name`` already exists, a ``ValueError`` will
   be raised, unless `overwrite=True` is specified.


:class:`JobDatabaseOutput`
---------------------------

:class:`JobOutput` provides access to the output of a single :term:`job`
executed during a WA :term:`run`, including metrics, artifacts, metadata, and
configuration stored in a postgres database.
The majority of attributes and methods are the same :class:`JobOutput` however the
noticeable differences are:

``basepath``
  A representation of the current database and host information backing this object.


methods
~~~~~~~

.. method:: JobDatabaseOutput.get_artifact(name)

    Return the :class:`Artifact` specified by ``name`` associated with this job.
    The `path` attribute of the :class:`Artifact` will be set to the Database
    OID of the object.

    :param name:  The name of the artifact to retrieve.
    :return: The :class:`Artifact` with that name
    :raises HostError: If the artifact with the specified name does not exist.

.. method:: JobDatabaseOutput.get_artifact_path(name)

    If the artifcat is a file this method returns a `StringIO` object containing
    the contents of the artifact specified by ``name`` associated with this job.
    If the aritifcat is a directory, the method returns a path to a locally
    extracted version of the directory which is left to the user to remove after
    use.

    :param name:  The name of the artifact who's path to retrieve.
    :return: A `StringIO` object with the contents of the artifact
    :raises HostError: If the artifact with the specified name does not exist.


:class:`Metric`
---------------

A metric represent a single numerical measurement/score collected as a result of
running the workload. It would be generated either by the workload or by one of
the augmentations active during the execution of the workload.

A :class:`Metric` has the following attributes:

``name``
    The name of the metric.

    .. note:: A name of the metric is not necessarily unique, even for the same
              job. Some workloads internally run multiple sub-tests, each
              generating a metric with the same name. In such cases,
              :term:`classifier`\ s are used to distinguish between them.

``value``
    The value of the metrics collected.


``units``
    The units of the metrics. This maybe ``None`` if the metric has no units.


``lower_is_better``
    The default assumption is that higher metric values are better. This may be
    overridden by setting this to ``True``, e.g. if metrics such as "run time"
    or "latency". WA does not use this internally (at the moment) but this may
    be used by external parties to sensibly process WA results in a generic way.


``classifiers``
    These can be user-defined :term:`classifier`\ s propagated from the job/run,
    or they may have been added by the workload to help distinguish between
    otherwise identical metrics.

``label``
    This is a string constructed from the name and classifiers, to provide a
    more unique identifier, e.g. for grouping values across iterations. The
    format is in the form ``name/cassifier1=value1/classifier2=value2/...``.


:class:`Artifact`
-----------------

An artifact is a file that is created on the host as part of executing a
workload. This could be trace, logging, raw output, or pretty much anything
else. Pretty much every file under WA output directory that is not already
represented by some other framework object will have an :class:`Artifact`
associated with it.

An :class:`Artifact` has  the following attributes:


``name``
    The name of this artifact. This will be unique for the job/run (unlike
    metric names). This is intended as a consistent "handle" for this artifact.
    The actual file name for the artifact may vary from job to job (e.g. some
    benchmarks that create files with results include timestamps in the file
    names), however the name will always be the same.

``path``
    Partial path to the file associated with this artifact. Often, this is just
    the file name. To get the complete path that maybe used to access the file,
    use :func:`get_artifact_path` of the corresponding output object.


``kind``
    Describes the nature of this artifact to facilitate generic processing.
    Possible kinds are:

    :log: A log file. Not part of the "output" as such but contains
            information about the run/workload execution that be useful for
            diagnostics/meta analysis.
    :meta: A file containing metadata. This is not part of the "output", but
            contains information that may be necessary to reproduce the
            results (contrast with ``log`` artifacts which are *not*
            necessary).
    :data: This file contains new data, not available otherwise and should
            be considered part of the "output" generated by WA. Most traces
            would fall into this category.
    :export: Exported version of results or some other artifact. This
                signifies that this artifact does not contain any new data
                that is not available elsewhere and that it may be safely
                discarded without losing information.
    :raw: Signifies that this is a raw dump/log that is normally processed
            to extract useful information and is then discarded. In a sense,
            it is the opposite of ``export``, but in general may also be
            discarded.

            .. note:: Whether a file is marked as ``log``/``data`` or ``raw``
                    depends on how important it is to preserve this file,
                    e.g. when archiving, vs how much space it takes up.
                    Unlike ``export`` artifacts which are (almost) always
                    ignored by other exporters as that would never result
                    in data loss, ``raw`` files *may* be processed by
                    exporters if they decided that the risk of losing
                    potentially (though unlikely) useful data is greater
                    than the time/space cost of handling the artifact (e.g.
                    a database uploader may choose to ignore ``raw``
                    artifacts, where as a network filer archiver may choose
                    to archive them).

    .. note:: The kind parameter is intended to represent the logical
              function of a particular artifact, not it's intended means of
              processing -- this is left entirely up to the output
              processors.

``description``
    This may be used by the artifact's creator to provide additional free-form
    information about the artifact. In practice, this is often ``None``


``classifiers``
    Job- and run-level :term:`classifier`\ s will be propagated to the artifact.


Additional run info
-------------------

:class:`RunOutput` object has ``target_info``  and ``run_info`` attributes that
contain structures that provide additional information about the run and device.

.. _target-info-api:

:class:`TargetInfo`
~~~~~~~~~~~~~~~~~~~

The :class:`TargetInfo` class presents various pieces of information about the
target device. An instance of this class will be instantiated and populated
automatically from the devlib `target
<http://devlib.readthedocs.io/en/latest/target.html>`_ created during a WA run
and serialized to a json file as part of the metadata exported
by WA at the end of a run.

The available attributes of the class are as follows:

``target``
    The name of the target class that was uised ot interact with the device
    during the run E.g.  ``"AndroidTarget"``, ``"LinuxTarget"`` etc.

``modules``
    A list of names of modules that have been loaded by the target. Modules
    provide additional functionality, such as access to ``cpufreq`` and which
    modules are installed may impact how much of the ``TargetInfo`` has been
    populated.

``cpus``
    A list of :class:`CpuInfo` objects describing the capabilities of each CPU.

``os``
    A generic name of the OS the target was running (e.g. ``"android"``).

``os_version``
    A dict that contains a mapping of OS version elements to their values. This
    mapping is OS-specific.

``abi``
    The ABI of the target device.

``hostname``
    The hostname of the the device the run was executed on.

``is_rooted``
    A boolean value specifying whether root was detected on the device.

``kernel_version``
    The version of the kernel on the target device.  This returns a
    :class:`KernelVersion` instance that has separate version and release
    fields.

``kernel_config``
    A :class:`KernelConfig` instance that contains parsed kernel config from the
    target device. This may be ``None`` if the kernel config could not be
    extracted.

``sched_features``
    A list of the available tweaks to the scheduler, if available from the
    device.

``hostid``
    The unique identifier of the particular device the WA run was executed on.


.. _run-info-api:

:class:`RunInfo`
~~~~~~~~~~~~~~~~

The :class:`RunInfo` provides general run information. It has the following
attributes:


``uuid``
    A unique identifier for that particular run.

``run_name``
    The name of the run (if provided)

``project``
    The name of the project the run belongs to (if provided)

``project_stage``
    The project stage the run is associated with (if provided)

``duration``
    The length of time the run took to complete.

``start_time``
    The time the run was stared.

``end_time``
    The time at which the run finished.
