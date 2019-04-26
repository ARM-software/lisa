.. _wltest-doc:

******
Wltest
******

Wltest enables the comparison of power and performance impacts of kernel
changes on Android devices. Wltest takes a list of kernel commits as input,
build and flashes each one on a device, and runs a `workload-automation` agenda
in order to collect power and performance metrics. After the execution, the
results can be parsed and analyized using the ipython agenda provided in Lisa
under ``ipynb/wltests/sched-evaluation-full.ipynb``

The ``lisa-wltest-series --help`` command provides the full list of parameters
for wltest.

Target types
============

Wltests supports two types of targets:

 1. the `standard` targets, for which the build procedure is aligned with
    upstream (using a simple defconfig and make command);

 2. the `repo` targets, for which the build procedure is entirely managed
    by a repo containing the kernel sources, modules, toolchains and build
    scripts. The repo targets have ``WLTEST_REPO_TARGET="y"`` set in their
    `definitions` file.

Standard targets
++++++++++++++++

For standard targets, such as Hikey 960, the wltest workflow goes as follows:

 1. Clone the kernel sources with
    ``git clone https://android.googlesource.com/kernel/hikey-linaro``

 2. Hack the kernel and commit your changes

 3. Repeat 2. as necessary

 4. Put in a 'series' file the list commits you want to test using the
    ``<sha1> <commit title>`` format as provided by
    ``git log --oneline --no-color --no-decorate``

 5. Run ``lisa-wltest-series`` with ``-k`` pointed at the kernel source tree and
    ``-s`` pointed at the file created in 4.

Repo targets
++++++++++++

The kernel sources of `repo` targets are managed using Google's ``repo`` tool
(see https://source.android.com/setup/develop/repo). The kernel repo usually
features the kernel sources, the module sources, the required toolchains, the
build scripts, and potentially more.

For repo targets, such as Pixel 3, the wltest workflow goes as follows:

 1. Download the sources using ``repo`` as explained in
    https://source.android.com/setup/build/building-kernels#downloading

 2. Issue ``repo start --all <branch name>`` in the repo

 3. Go in the sub-trees (kernel and/or modules), hack the code, and commit
    your changes

 4. Repeat 2. and 3. as necessary

 5. Put in a 'series' files the list of repo branches you want to test

 6. Run ``lisa-wltest-series`` with ``-k`` pointed at the top level repo folder
    downloaded in 1., and ``-s`` pointing at the file created in 5.
