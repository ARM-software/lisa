execution_order:  
    type: ``'str'``

    Defines the order in which the agenda spec will be executed. At the
    moment, the following execution orders are supported:

    ``"by_iteration"``
        The first iteration of each workload spec is executed one after
        the other, so all workloads are executed before proceeding on
        to the second iteration.  E.g. A1 B1 C1 A2 C2 A3. This is the
        default if no order is explicitly specified.

        In case of multiple sections, this will spread them out, such
        that specs from the same section are further part. E.g. given
        sections X and Y, global specs A and B, and two iterations,
        this will run ::

                X.A1, Y.A1, X.B1, Y.B1, X.A2, Y.A2, X.B2, Y.B2

    ``"by_section"``
        Same  as ``"by_iteration"``, however this will group specs from
        the same section together, so given sections X and Y, global
        specs A and B, and two iterations, this will run ::

                X.A1, X.B1, Y.A1, Y.B1, X.A2, X.B2, Y.A2, Y.B2

    ``"by_spec"``
        All iterations of the first spec are executed before moving on
        to the next spec. E.g. A1 A2 A3 B1 C1 C2.

    ``"random"``
        Execution order is entirely random.

    allowed values: ``'by_iteration'``, ``'by_spec'``, ``'by_section'``, ``'random'``

    default: ``'by_iteration'``

reboot_policy:  
    type: ``'RebootPolicy'``

    This defines when during execution of a run the Device will be
    rebooted. The possible values are:

    ``"as_needed"``
        The device will only be rebooted if the need arises (e.g. if it
        becomes unresponsive.

    ``"never"``
        The device will never be rebooted.

    ``"initial"``
        The device will be rebooted when the execution first starts,
        just before executing the first workload spec.

    ``"each_spec"``
        The device will be rebooted before running a new workload spec.

        .. note:: this acts the same as each_iteration when execution order
                  is set to by_iteration

    ``"each_iteration"``
        The device will be rebooted before each new iteration.

    allowed values: ``'never'``, ``'as_needed'``, ``'initial'``, ``'each_spec'``, ``'each_iteration'``

    default: ``'as_needed'``

device:  
    type: ``'str'``

    This setting defines what specific Device subclass will be used to
    interact the connected device. Obviously, this must match your
    setup.

    default: ``'generic_android'``

retry_on_status:  
    type: ``'list_of_Enums'``

    This is list of statuses on which a job will be considered to have
    failed and will be automatically retried up to ``max_retries``
    times. This defaults to ``["FAILED", "PARTIAL"]`` if not set.
    Possible values are:

    ``"OK"``
        This iteration has completed and no errors have been detected

    ``"PARTIAL"``
        One or more instruments have failed (the iteration may still be
        running).

    ``"FAILED"``
        The workload itself has failed.

    ``"ABORTED"``
        The user interrupted the workload.

    allowed values: ``RUNNING``, ``OK``, ``PARTIAL``, ``FAILED``, ``ABORTED``, ``SKIPPED``

    default: ``['FAILED', 'PARTIAL']``

max_retries:  
    type: ``'integer'``

    The maximum number of times failed jobs will be retried before
    giving up. If not set.

    .. note:: this number does not include the original attempt

    default: ``2``

bail_on_init_failure:  
    type: ``'boolean'``

    When jobs fail during their main setup and run phases, WA will
    continue attempting to run the remaining jobs. However, by default,
    if they fail during their early initialization phase, the entire run
    will end without continuing to run jobs. Setting this to ``False``
    means that WA will instead skip all the jobs from the job spec that
    failed, but continue attempting to run others.

    default: ``True``

allow_phone_home:  
    type: ``'boolean'``

    Setting this to ``False`` prevents running any workloads that are marked
    with 'phones_home', meaning they are at risk of exposing information
    about the device to the outside world. For example, some benchmark
    applications upload device data to a database owned by the
    maintainers.

    This can be used to minimise the risk of accidentally running such
    workloads when testing confidential devices.

    default: ``True``

