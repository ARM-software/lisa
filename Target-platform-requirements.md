## Contents

1. [Linux Targets](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#linux-targets)
2. [Android Targets](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#android-targets)
3. [Kernel features](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#kernel-features)

The target platform to be used for experiments with LISA must satisfy
the following requirements:

## Linux Targets

- allow `ssh` access, preferably as root, using either a password or an SSH key.

  Note that on Ubuntu targets, SSH root access can be enabled by setting `PermitRootLogin yes`
  in the file `/etc/ssh/sshd_config`, and then restarting the SSH daemon.

- support `sudo`, even if it's accessed as root user

## Android Targets

- allow `adb` access, eventually by specifying a *DEVICE ID*
- the local shell should define the `ANDROID_HOME` environment variable pointing
  to an Android SDK installation. If you are not using the virtual machine based
  installation you will have to install the command line tools from
  [here](https://developer.android.com/studio/index.html).

Analysis on Android devices can be done using `systrace` instead of `ftrace`. If
you plan to use `systrace` read
[this guide](https://github.com/ARM-software/lisa/wiki/Android-Tools-for-Tracing).

## Kernel features

Most of the tests targets a kernel with support for some new frameworks which
are currently in-development:

- Energy-Aware Scheduler (EAS)
- SchedFreq: the CPUFreq governor
- SchedTune: the central, scheduler-driven, power-perfomance control

Tests targeting an evaluation of these frameworks requires also a set of
tracepoint which are not available in mainline kernel. The series of patches
required to add to a recent kernel the tracepoints required by some tests are
available on this git repository and branch:

	$ git://www.linux-arm.org/linux-pb.git lisa/debug

The patches required are the ones in this series:

	$ git log --oneline lisa/debug_base..lisa/debug

There is also a [GitWeb Link](http://www.linux-arm.org/git?p=linux-pb.git;a=shortlog;h=refs/heads/lisa/debug) to the list of required tracepoints which are the topmost patches with a name starting by "DEBUG: ".
