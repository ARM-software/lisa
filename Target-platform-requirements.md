## Contents

1. [Linux Targets](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#linux-targets)
2. [Android Targets](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#android-targets)
3. [Kernel features](https://github.com/ARM-software/lisa/wiki/Target-platform-requirements#kernel-features)

The target platform to be used for experiments with LISA must satisfy
the following requirements:

## Linux Targets

- allow *ssh* access, preferably as root, using either a password or an SSH key
- support *sudo*, even if it's accessed as root user

## Android Targets

- allow *adb* access, eventually by specifying a DEVICE ID
- the local shell should define the *ANDROID_HOME* environment variable pointing
  to an Android SDK installation

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