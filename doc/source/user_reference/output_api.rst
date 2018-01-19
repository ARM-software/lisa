.. _output-api:

Output API
==========

WA3 now has an output API that can be used to post process a run's
:ref:`Output Directory Structure<output_directory>` which can be performed by using WA's
``RunOutput`` object.

Example:

If we have an existing WA output called ``wa_output`` in the current working
directory we can initialize a ``RunOutput`` as follows:

.. code-block:: python

    In [1]: from wa import RunOutput
       ...:
       ...: output_folder = 'wa_output'
       ...: run_output = RunOutput(output_folder)



From here we can retrieve different information about the run. For example if we
want to see what the status of the run was and retrieve the metrics recorded from
the first run  we can do the following:

.. code-block:: python

    In [2]: run_output.status
    Out[2]: OK(7)

    In [3]: run_output.jobs
    Out[3]:
    [<wa.framework.output.JobOutput at 0x7f70358a1f10>,
     <wa.framework.output.JobOutput at 0x7f70358a1150>,
     <wa.framework.output.JobOutput at 0x7f7035862810>,
     <wa.framework.output.JobOutput at 0x7f7035875090>]

    In [4]: job_1 = run_output.jobs[0]

    In [5]: job_1.label
    Out[5]: u'dhrystone'

    In [6]: job_1.metrics
    Out[6]:
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


We can also retrieve information about the device that the run was performed on:

.. code-block:: python

    In [7]: run_output.target_info.os
    Out[7]: u'android'

    In [8]: run_output.target_info.os_version
    Out[8]:
    OrderedDict([(u'all_codenames', u'REL'),
                 (u'incremental', u'3687331'),
                 (u'preview_sdk', u'0'),
                 (u'base_os', u''),
                 (u'release', u'7.1.1'),
                 (u'codename', u'REL'),
                 (u'security_patch', u'2017-03-05'),
                 (u'sdk', u'25')])


