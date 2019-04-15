Man page
========

DESCRIPTION
+++++++++++

``bisector`` is a ``git bisect run`` [#]_ compatible tool used in LISA. Its goal is
to sequence commands to be repeated an arbitrary number of times and generates
a report.

For example, ``bisector`` can be used to compile a kernel, flash a board,
reboot it and run test suite. To achieve that, ``bisector`` runs a sequence of
steps, and saves a report file containing the result of each step. This step
sequence is run for a number of iterations, or for a given duration. That
allows tracking fluctuating bugs by repeating a test process many times.

The steps class can implement any run or report behavior, with the ability to
take parameters. They also decide what data is saved into the report, and
their contribution to the overall ``git bisect`` result:

* `good`: the test steps passed
* `bad`: the test steps failed
* `untestable`: the build or flash step failed 
* `abort`: the reboot step failed, and the board is now unusable and requires
  manual intervention
* `NA`: the step will not impact the overall result

It is important to note that ``bisector`` step classes usually only invoke a
user-specified command line tool and will use the return code to decide what
``git bisect`` result to return. They can also implement more specific result
reporting, and additional run behaviors.


OPTIONS
+++++++

bisector
--------

.. run-command::
   :ignore-error:
   :literal:

   bisector --help

bisector run
------------

.. run-command::
   :ignore-error:
   :literal:

   bisector run --help

bisector report
---------------

.. run-command::
   :ignore-error:
   :literal:

   bisector report --help

bisector step-help
------------------

Steps-specific options to be used with ``bisector run -o`` and ``bisector report -o``.

.. run-command::
   :ignore-error:
   :literal:

   bisector step-help

bisector monitor
----------------

.. run-command::
   :ignore-error:
   :literal:

   bisector monitor --help

bisector monitor-server
-----------------------

.. run-command::
   :ignore-error:
   :literal:

   bisector monitor-server --help

bisector edit
-------------

.. run-command::
   :ignore-error:
   :literal:

   bisector edit --help

CONFIGURATION
+++++++++++++

``bisector run`` is configured using a YAML [#]_ file specified using ``--steps`` that
defines the steps that will be executed in a loop. Each declared step has a (usually
unique) name and a class that will influence the way its result is used and its
options.

The YAML file is structured as following:

.. code-block:: yaml

   # Top-level "steps" key is important as the same file can be used to host other
   # information.
   steps:
      
      # build step will interpret a non-zero exit status of the command as a
      # bisect untestable status, and zero exit status as bisect good.
       -
         class: BuildStep
         cmd: make defconfig Image dtbs
         trials: 1

      # flash step will interpret a non-zero exit status of the command as a
      # bisect abort, and zero exit status ignored.
      -
         class: FlashStep
         cmd: flash-my-board
         timeout: 180
         trials: 5

      # reboot step will interpret a non-zero exit status of the command as a
      # bisect abort, and zero exit status as bisect good.
       - 
         class: reboot
         timeout: 300
         trials: 10
         cmd: reboot-my-board

       # exekall LISA test will interpret a non-zero exit status of the command
       # as bisect bad, and a zero exit status as bisect good.
       -
         class: LISA-test
         name: one-small-task
         # Using systemd-run ensures all child process is killed if the session
         # is interrupted
         use-systemd-run: true
         timeout: 3600
         cmd: lisa-test 'OneSmallTask*'

All step options can also be specified using ``--options/-o``, which will
override what is described in the YAML steps configuration.

MONITORING
++++++++++

``bisector run`` allows some live monitoring by exposing a DBus interface. This
is used by two subcommands: ``bisector monitor-server`` and ``bisector
monitor``.

Server
------

``bisector monitor-server`` acts as a registry of all ``bisector run``
executing under the same user (DBus session bus). It allows ``bisector
monitor`` to list active instances and also forwards desktop notifications to
the desktop environment. The server can be (re)started after ``bisector run``
if necessary.

Monitor
-------

``bisector monitor`` allows listing active instances of ``bisector run`` (when
the server is running), and allows querying various information from them.
Since the query can be directed to a specific PID, the server is only necessary
for listing.

EXAMPLES
++++++++

A typical flow of ``bisector`` looks like that:

.. code-block:: shell

   # Run the steps and generate a report.
   # systemd-run will be used for all steps by using "-o" without specifying a
   step name or category.
   bisector run --desc "description of my report" --steps bisector_steps.yaml --report bisector.report.yml.gz -ouse-systemd-run=yes

   # Later inspection of the report, only looking at the steps that have "test"
   # name or category.
   bisector report bisector.report.yml.gz
    
   # Show steps with the "test" name or category
   bisector report bisector.report.yml.gz --only test
    
   # Help of a exekall LISA's step options
   bisector step-help LISA-test
    
   # Get all information about tests failures
   bisector report bisector.report.yml.gz -overbose
    
   # Show the tests failure backtraces and messages, and the metrics in the
   # other cases.
   # -oshow-details=msg only displays the message without the backtrace
   bisector report bisector.report.yml.gz -oshow-details
    
   # Ignore some exceptions in LISA results
   # These exceptions are related to network/ssh issues and are usually not interesting
   bisector report bisector.report.yml.gz -oignore-excep=ExceptionPxssh,HostError,TimeoutError
    
   # Only show the results of a specific test case
   # the name is as reported by exekall, so it is best to use * to match the
   # boilerplate prefix.
   # For example -otestcase='OneSmallTask:*' will match both
   # "OneSmallTask:test_slack" and "OneSmallTask:test_task_placement".
   bisector report bisector.report.yml.gz -otestcase='OneSmallTask:*'
    
   # Only consider iterations 1 to 5, 42 and 56
   # Useful to limit the amount of downloaded result archives
   bisector report bisector.report.yml.gz -oiterations=1-5,42,56
    
   # Ignore LISA tests results that are not failures
   # use that to only download the result archives for failed tests.
   bisector report bisector.report.yml.gz -oignore-non-issue
    
   # Download the archives of failed tests and export stdout/sdterr logs to files in
   # the "logs" directory.
   # The hierarchy created is <folder to export to>/<step name>/<iteration number>
   # This will create files for commands output, xUnit files and download the result archives.
   bisector report bisector.report.yml.gz -oexport-logs=logs
    
   # "-oXXXX=YYYY" options can be applied to a specific step instead of all of them using
   # -o <step name or category>.XXXX=YYYY
   # This command will only show iteration #2 for the eas_behaviour step
   bisector report bisector.report.yml.gz -oeas_behaviour.iterations=2


REFERENCES
++++++++++

.. [#] https://git-scm.com/docs/git-bisect#_bisect_run
.. [#] https://learnxinyminutes.com/docs/yaml/
