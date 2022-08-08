.. _revent_files_creation:

Automating GUI Interactions With Revent
=======================================

Overview and Usage
------------------

The revent utility can be used to record and later play back a sequence of user
input events, such as key presses and touch screen taps. This is an alternative
to Android UI Automator for providing automation for workloads.

Using revent with workloads
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some workloads (pretty much all games) rely on recorded revents for their
execution. ReventWorkloads require between 1 and 4 revent files to be ran.
There is one mandatory recording, ``run``, for performing the actual execution of
the workload and the remaining stages are optional. ``setup`` can be used to perform
the initial setup (navigating menus, selecting game modes, etc).
``extract_results`` can be used to perform any actions after the main stage of
the workload for example to navigate a results or summary screen of the app. And
finally ``teardown`` can be used to perform any final actions for example
exiting the app.

Because revents are very device-specific\ [*]_, these files would need to
be recorded for each device.

The files must be called ``<device name>.(setup|run|extract_results|teardown).revent``,
where ``<device name>`` is the name of your device (as defined by the model
name of your device which can be retrieved with
``adb shell getprop ro.product.model`` or by the ``name`` attribute of your
customized device class).

WA will look for these files in two places:
``<installdir>/wa/workloads/<workload name>/revent_files`` and
``$WA_USER_DIRECTORY/dependencies/<workload name>``. The
first location is primarily intended for revent files that come with WA (and if
you did a system-wide install, you'll need sudo to add files there), so it's
probably easier to use the second location for the files you record. Also, if
revent files for a workload exist in both locations, the files under
``$WA_USER_DIRECTORY/dependencies`` will be used in favour
of those installed with WA.

.. [*] It's not just about screen resolution -- the event codes may be different
       even if devices use the same screen.

.. _revent-recording:

Recording
^^^^^^^^^

WA features a ``record`` command that will automatically deploy and start revent
on the target device.

If you want to simply record a single recording on the device then the following
command can be used which will save the recording in the current directory::

    wa record

There is one mandatory stage called 'run' and 3 optional stages: 'setup',
'extract_results' and 'teardown' which are used for playback of a workload.
The different stages are distinguished by the suffix in the recording file path.
In order to facilitate in creating these recordings you can specify ``--setup``,
``--extract-results``, ``--teardown`` or ``--all`` to indicate which stages you
would like to create recordings for and the appropriate file name will be generated.

You can also directly specify a workload to create recordings for and WA will
walk you through the relevant steps. For example if we waned to create
recordings for the Angrybirds Rio workload we can specify the ``workload`` flag
with ``-w``. And in this case WA can be used to automatically deploy and launch
the workload and record ``setup`` (``-s``) , ``run`` (``-r``) and ``teardown``
(``-t``) stages for the workload. In order to do this we would use the following
command with an example output shown below::

    wa record -srt -w angrybirds_rio

::

    INFO     Setting up target
    INFO     Deploying angrybirds_rio
    INFO     Press Enter when you are ready to record SETUP...
    [Pressed Enter]
    INFO     Press Enter when you have finished recording SETUP...
    [Pressed Enter]
    INFO     Pulling '<device_model>setup.revent' from device
    INFO     Press Enter when you are ready to record RUN...
    [Pressed Enter]
    INFO     Press Enter when you have finished recording RUN...
    [Pressed Enter]
    INFO     Pulling '<device_model>.run.revent' from device
    INFO     Press Enter when you are ready to record TEARDOWN...
    [Pressed Enter]
    INFO     Press Enter when you have finished recording TEARDOWN...
    [Pressed Enter]
    INFO     Pulling '<device_model>.teardown.revent' from device
    INFO     Tearing down angrybirds_rio
    INFO     Recording(s) are available at: '$WA_USER_DIRECTORY/dependencies/angrybirds_rio/revent_files'

Once you have made your desired recordings, you can either manually playback
individual recordings using the :ref:`replay <replay-command>` command or, with
the recordings in the appropriate dependencies location, simply run the workload
using the :ref:`run <run-command>` command and then all the available recordings will be
played back automatically.

For more information on available arguments please see the :ref:`Record <record_command>`
command.

    .. note:: By default revent recordings are not portable across devices and
              therefore will require recording for each new device you wish to use the
              workload on. Alternatively a "gamepad" recording mode is also supported.
              This mode requires a gamepad to be connected to the device when recording
              but the recordings produced in this mode should be portable across devices.

.. _revent_replaying:

Replaying
^^^^^^^^^

If you want to replay a single recorded file, you can use ``wa replay``
providing it with the file you want to replay. An example of the command output
is shown below::

        wa replay my_recording.revent
        INFO     Setting up target
        INFO     Pushing file to target
        INFO     Starting replay
        INFO     Finished replay

If you are using a device that supports android you can optionally specify a
package name to launch before replaying the recording.

If you have recorded the required files for your workload and have placed the in
the appropriate location (or specified the workload during recording) then you
can simply run the relevant workload and your recordings will be replayed at the
appropriate times automatically.

For more information run please read :ref:`replay-command`

Revent vs UiAutomator
----------------------

In general, Android UI Automator is the preferred way of automating user input
for Android workloads because, unlike revent, UI Automator does not depend on a
particular screen resolution, and so is more portable across different devices.
It also gives better control and can potentially be faster for doing UI
manipulations, as input events are scripted based on the available UI elements,
rather than generated by human input.

On the other hand, revent can be used to manipulate pretty much any workload,
where as UI Automator only works for Android UI elements (such as text boxes or
radio buttons), which makes the latter useless for things like games. Recording
revent sequence is also faster than writing automation code (on the other hand,
one would need maintain a different revent log for each screen resolution).

.. note:: For ChromeOS targets, UI Automator can only be used with android
          applications and not the ChomeOS host applications themselves.


