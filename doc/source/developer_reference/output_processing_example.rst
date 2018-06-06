
Output Processing API Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To demonstrate how we can use the output API if we have an existing WA output
called ``wa_output`` in the current working directory we can initialize a
``RunOutput`` as follows:

.. code-block:: python

    In [1]: from wa import RunOutput
       ...:
       ...: output_directory = 'wa_output'
       ...: run_output = RunOutput(output_directory)



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
    In [6]: for k, v in job_1.spec.runtime_parameters.iteritems():
       ...:     print (k, v)
    (u'airplane_mode': False)
    (u'brightness': 100)
    (u'governor': 'userspace')
    (u'big_frequency': 1700000)
    (u'little_frequency': 1400000)

    # Print out all the metrics avalible for this job
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


