.. _setup-page:

*****
Setup
*****

Installation
============

Host installation
+++++++++++++++++

From PyPI
---------
.. _setup-pypi:


LISA is available on `PyPI <https://pypi.org/project/lisa-linux/>`_:

.. code:: shell

   pip install lisa-linux

.. note:: Some dependencies cannot be fulfilled by PyPI, such as ``adb`` when
    working with Android devices. It is the user's responsability to install
    them. Alternatively, the installation from the git repository allows setting
    up a full environment.

From GitHub
-----------

LISA is hosted at `github <https://github.com/ARM-software/lisa>`_.
The following references are available:

    * ``master`` branch: Main development branch where pull requests are merged as they
      come.
    * ``release`` branch: Branch updated upon release of the ``lisa-linux`` package on
      PyPI.
    * ``vX.Y.Z`` tags: One tag per release of ``lisa-linux`` PyPI package.


LISA has a minimum Python version requirement of:

.. exec::
    import os
    from lisa._git import find_root
    cwd = os.getcwd()
    root = find_root('.')
    try:
        os.chdir(root)
        import setup
        print('Python', setup.python_requires)
    finally:
        os.chdir(cwd)

If your distribution does not natively ship with a high-enough version, you can
install it manually and provide the name of the Python binary before doing any
other action:

.. code:: shell

    export LISA_PYTHON=<name of Python 3 binary>

On Ubuntu, the ``deadsnakes`` PPA provides alternative versions of Python. Note
that Ubuntu splits the Python distribution into multiple packages, which must
all be installed. The list is available inside ``install_base.sh`` and is more
or less:

    * python3
    * python3-pip
    * python3-venv
    * python3-tk

You might also find a ``python3.X-full`` package that contains everything you
need.

.. code:: shell

    git clone https://github.com/ARM-software/lisa.git
    # This will provide a more accurate changelog when building the doc
    git fetch origin refs/notes/changelog
    cd lisa
    # A few packages need to be installed, like python3 or kernelshark. Python
    # modules will be installed in a venv at the next step, without touching
    # any system-wide install location.
    sudo ./install_base.sh --install-all
    # On the first run, it will take care of creating a Python venv and populating it
    source init_env

.. attention:: If you use this installation procedure, make sure to always run
    ``source init_env`` before anything else in order to activate the venv.
    Otherwise, importing lisa will fail, commands will not be available etc.

.. attention:: Only Bash and ZSH are officially supported by ``source
    init_env``. Mileage may vary for other shells, up to and including failure
    from the first line.


In case the venv becomes unusable for some reason, the ``lisa-install``
shell command available after sourcing ``init_env`` will allow to create a new
clean venv from scratch.

Additional Python packages
..........................

``lisa-install`` will also install the content of
``$LISA_HOME/custom_requirements.txt`` if the file exists. That allows
re-installing a custom set of packages automatically when the venv needs to
regenerated.

Without automatic ``venv``
..........................

Sometimes, LISA needs to operate in an environment setup for multiple tools. In
that case, it may be easier to manage manually a venv/virtualenv instead of
letting LISA create one for its shell.

Setting ``export LISA_USE_VENV=0`` prior to ``source init_env`` will avoid the
creation and usage of the LISA-managed venv. ``lisa-install`` command can still
be used to install the necessary Python packages, which will honor any
venv-like system manually setup.

Alternatively, ``lisa`` package is packaged according to the usual Python
practices, which includes a ``setup.py`` script, and a
``devmode_requirements.txt`` file that will install all the shipped packages in
editable mode (including those that are not developped in that repository, but
still included for convenience).

Virtual machine installation
----------------------------
.. _setup-vagrant:

LISA provides a Vagrant recipe which automates the generation of a
VirtualBox based virtual machine pre-configured to run LISA. To generate and
use such a virtual machine you need:

- `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`__
- `Vagrant <https://www.vagrantup.com/downloads.html>`__

Once these two components are available on your machine, issue these commands:

.. code:: shell

  git clone https://github.com/ARM-software/lisa.git
  cd lisa
  vagrant up

This last command builds and executes the VM according to the description provided
by the Vagrant file available in the root folder of the LISA source tree.

Once the VM installation is complete, you can access that VM with:

.. code:: shell

  vagrant ssh

.. important:: In order to work around a
  `Vagrant bug <https://github.com/hashicorp/vagrant/issues/12057>`_, all the
  dependencies of LISA are installed in non-editable mode inside the VM. This
  means that using `git pull` must be followed by a `lisa-install` if any of the
  dependencies in `external/` are updated.


Target installation
+++++++++++++++++++

LISA's "device under test" is called target. In order to be able to run e.g.
tests on a target, you will need the provide a minimal environment composed of:

    * An ``adb`` or ``ssh`` server
    * For some tests, a working Python 3 installation

This can be provided by a a regular GNU/Linux or Android distribution, but can
also be done with a minimal buildroot environment. The benefits are:

    * Almost no background task that can create issues when testing the Linux
      kernel scheduler
    * Can be used as a in-memory initramfs, thereby avoiding activity of USB or
      NFS-related kthreads, as it has been the source of issues on some boards
      with wonky USB support.
    * Using initramfs has the added advantages of ease of deployment (can be
      integrated in the kernel image, reducing the amount of assets to flash)
      and avoids issues related to board state (a reboot fully resets the
      userspace).

Buildroot image creation is assisted with these commands, available in lisa
shell :ref:`buildroot-commands`.


Kernel modules
--------------

The following modules are required to run Lisa tests against some kernels.

sched_tp
........

From Linux v5.3, sched_load_cfs_rq and sched_load_se tracepoints are present in
mainline as bare tracepoints without any events in tracefs associated with
them.

To help expose these tracepoints (and any additional one we might require in
the future) as trace events, an external module is required and is provided
under the name of sched_tp in $LISA_HOME/tools/kmodules/sched_tp

Building a module
-----------------

LISA Python package will compile and load the module automatically when required
for tracing so there is usually no reason to do so manually. The most reliable
way to configure LISA for building the module is:

  * Kernel config:

    .. code-block:: sh

      CONFIG_IKHEADERS=y
      CONFIG_IKCONFIG=y
      CONFIG_DEBUG_INFO=y
      CONFIG_DEBUG_INFO_BTF=y
      CONFIG_DEBUG_INFO_REDUCED=n
      CONFIG_BPF_SYSCALL=y

  * Target configuration (:class:`lisa.target.TargetConf`):

    .. code-block:: yaml

      target-conf:
          kernel:
              # If this is omitted, LISA will try to download a kernel.org
              # released tarball. If the kernel has only minor differences with
              # upstream, it will work, but can also result in compilation
              # errors due to mismatching headers.
              src: /home/foobar/linux/
              modules:
                  # This is not mandatory but will use a tested chroot to build
                  # the module. If that is omitted, ``CROSS_COMPILE`` will be
                  # used (and inferred if not set).
                  build-env: alpine

                  # It is advised not to set that, but in case overlayfs is
                  # unusable (e.g. inside an LXC or docker container for a CI
                  # system depending on config), this should do the trick.
                  # overlay-backend: copy

In case this is still required, the process is standard Linux external module
build step. Helper scripts are provides too.

Build
.....

.. code-block:: sh

  $LISA_HOME/tools/kmodules/build_module path/to/kernel path/to/kmodule [path/to/install/modules]

This will build the module against the provided kernel tree and install it in
``path/to/install/module`` if provided otherwise install it in
``$LISA_HOME/tools/kmodules``.

.. warning:: The documentation used to refer to
  ``$LISA_HOME/lisa/_assets/kmodules`` rather than
  ``tools/kmodules``. This was an oversight, DO NOT build from
  ``lisa/_assets``. If you still do, any remaining build artifact
  could be reused in fresh builds, leading to segfaults and such.

Clean
.....

.. code-block:: sh

  $LISA_HOME/tools/kmodules/clean_module path/to/kernel path/to/kmodule

Highly recommended to clean when switching kernel trees to avoid unintentional
breakage for using stale binaries.

Pushing the module into the target
..................................

You need to push the module into your rootfs either by installing it directly
there or use commands like ``scp`` to copy it into your device.

.. code-block:: sh

  scp -r /path/to/sched_tp.ko username@ip:/

Loading the module
..................

On the target run:

.. code-block:: sh

  modprobe sched_tp

Integrating the module in your kernel tree
------------------------------------------

If you're rebuilding your kernel tree anyway, it might be easier to integrate
the module into your kernel tree as a built-in module so that it's always
present.

Updating
========

Over time, we might change/add some dependencies to LISA. As such, if you
update your LISA repository, you should make sure your locally-installed
packages still match those dependencies. Sourcing ``init_env`` from a
new shell should suffice, which will hint the user if running ``lisa-install``
again is needed.

.. note:: LISA does not provide any specific mean of keeping a venv up-to-date.
    Running ``lisa-install`` will destroy the venv it create and create a new
    one afresh, but doing so is the sole responsibility of the user, it will not
    happen automatically based on releases of new versions of LISA's
    dependencies.


What next ?
===========

The next step depends on the intended use case, further information at
:ref:`workflows-page`
