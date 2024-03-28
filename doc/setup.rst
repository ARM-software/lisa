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

From GitLab
-----------

LISA is hosted on `GitLab <https://gitlab.arm.com/tooling/lisa>`_.
The following references are available:

    * ``main`` branch: Main development branch where pull requests are merged as they
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

    git clone https://gitlab.arm.com/tooling/lisa
    # Jump into the cloned repo directory
    cd lisa
    # This will provide a more accurate changelog when building the doc
    git fetch origin refs/notes/changelog
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

  git clone https://gitlab.arm.com/tooling/lisa
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

From Linux v5.3, sched_load_cfs_rq and sched_load_se tracepoints are present in
mainline as bare tracepoints without any events in tracefs associated with
them.

To help expose these tracepoints (and any additional one we might require in
the future) as trace events, an external module is required and is provided
under the name of "lisa" in $LISA_HOME/tools/kmodules/lisa

Pre-requisites
..............

CFI
~~~

Using the out-of-tree build method for kernels with CONFIG_CFI_CLANG=y as all
Android kernels come by default requires the module to be built with at least
clang-16. This can either be achieved by using the ``alpine`` build environment,
by having it installed on host and using ``LLVM=1`` or forcing the version with
``LLVM=-16`` in ``target-conf/kernel/modules/make-variables``.

Kernel symbols needed for reading files on Android product kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use some Lisa module features (e.g. the lisa__pixel6_emeter ftrace event)
on a product kernel, some symbols forbidden by Google need to be re-enabled.

In order to do that, the kernel will need to be built with:

.. code-block:: sh

    ./update_symbol_list.sh

The script should be included in the product kernel tree. It will ensure that the required
symbols are not stripped from the final kernel image and the module does not get rejected.

Enabling a module
.................

LISA Python package will compile and load the module automatically when required
for tracing so there is usually no reason to do so manually. The most reliable
way to configure LISA for building the module is:

  * Kernel config (also available under ``$LISA_HOME/tools/kmodules/kconfig_fragment.config``):

    .. exec::
       :literal:

        from pathlib import Path
        from lisa._assets import ASSETS_PATH
        frag_path = Path(ASSETS_PATH) / 'kmodules' / 'kconfig_fragment.config'
        frag = frag_path.read_text()
        print(frag)

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

.. note:: If ``build-env: host`` is used (default), ensure that your setup is
    ready to compile a kernel. Notably, ensure that you have kernel build
    dependencies installed. This can be achieved with
    ``install_base.sh --install-kernel-build-dependencies`` (included in
    ``--install-all``)

Automatic route
...............

Once the kernel and LISA's target have been configured appropriately, the Python
API will build and load the module automatically as required (e.g. when ftrace
events provided by the module are required).

In order to improve interoperation with other systems, a CLI tool is also
provided to load the module easily:

  .. code-block:: sh

    # Compile and load the module.
    lisa-load-kmod --conf target_conf.yml

    # Runs "echo hello world" with the module loaded, then unloads it.
    lisa-load-kmod --conf target_conf.yml -- echo hello world

    # See # lisa-load-kmod --help for more options.


.. note:: The module name may be different if it was compiled manually vs
    compiled via the Python interface due to backward compatiblity
    constraints.


Manual route
............
  .. _manual-module-setup-warning:
  .. _manual-module-setup-warning2:

Manual build of the module are not supported. You may be able to hack your way
but if you do so, you are on your own. Also keep in mind that you will need to
re-implement internal mechanisms of LISA that might change at any time, so you
will loose any backward compatibility guarantee.

.. This is not supported anymore, and also not necessary these days.
..
  As a last resort option, the module can be built manually. Be aware that the
  automatic route is applying a number of workarounds you might have to discover
  and replicate yourself.

  .. _manual-module-setup-warning:
  .. warning::

    There is also no stability guarantee on any of the interfaces exposed by the
    module, such as it's CLI parameters. The behavior of enabling all features by
    default might also change, as well as the way of selecting features. The fact
    that all features are compiled-in and available is also not a given and might
    change in the future, making a specific build more tailored to a specific use
    case.

  However, there is sometimes no other choice, and this might still be useful as a
  temporary workaround. Just bear in mind that doing that will force you to
  monitor more closely what is happening in LISA, and gain more knowledge of its
  internal mechanisms to keep the setup working.

  .. _manual-module-setup-warning2:
  .. warning::

    If you share this setup with anyone else, it is your responsibility to
    forward the appropriate documentation pointers and maintenance knowledge, and
    most importantly to let them know what they are signing up for. It is also
    your responsibility to assert whether it makes sense for them to embark on
    that path. Things will break, whoever you share it with will complain (to
    you) if you have not appropriately made them aware of the situation. You have
    been warned.

Build
~~~~~

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
~~~~~

.. code-block:: sh

  $LISA_HOME/tools/kmodules/clean_module path/to/kernel path/to/kmodule

Highly recommended to clean when switching kernel trees to avoid unintentional
breakage for using stale binaries.

Integrating the module in your kernel tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method is not supported. It falls under the category of manual module
build.

.. This block is commented out as it will not work as it stand. If we were to
   resurrect that flow, it would be a good starting point.
..
    If you're rebuilding your kernel tree anyway, it might be easier to integrate
    the module into your kernel tree as a built-in module so that it's always
    present.

    .. warning::
      This method is less supported than the out-of-tree method above. It also has
      all the drawbacks of manual build root since it qualifies as manually
      building the module.

    In order to do that, follow the steps below:

    * Disable Google's ABI symbols checks by applying the patch found under
      ``tools/kmodules/lisa-in-tree/android/abi`` to the tree in ``build/abi``.

    * Apply the patches in ``tools/kmodules/lisa-in-tree/linux``
      to include a stub Kbuild Makefile structure for the module.
      For Android product kernels it should be applied under ``private/gs-google``,
      for Android mainline kernels under ``common``.

    .. note:: Older Android product kernels might be missing some internal header
      import guards present in newer mainline versions. For this method to work
      make sure your kernel tree includes mainline commits 95458477f5b2dc436e3aa6aa25c0f84bb83e6195
      and d90a2f160a1cd9a1745896c381afdf8d2812fd6b.

    * Additionally, on Android kernels it can be useful to apply the patches in
      ``tools/kmodules/lisa-in-tree/android`` as well. It will include the module
      in the vendor modules list for Android so that it is automatically loaded
      at boot-time. The patch is specific to the Pixel 6 source tree
      and very likely should be adjusted accordingly for any other platform.

    * Then, put the script found under ``tools/kmodules/lisa-in-tree/fetch_lisa_module.py``
      and follow the instructions in ``--help`` to link or fetch the Lisa module sources into
      the source tree.

    .. code-block:: sh

      ./fetch_lisa_module.py --module-kernel-path ./private/gs-google/drivers/soc/arm/vh/kernel/lisa --git-ref main

    With all these steps complete, rebuild the kernel:

    .. code-block:: sh

        ./update_symbol_list.sh

    The module should be built in-tree and then loaded at boot-time.

    .. note:: The order at which the module is loaded at boot time is not guaranteed and
      Android will not perform any of the Lisa module setup steps. Usually e.g. ``pixel6_emeter``
      will fail to load on boot and the module will have to be reloaded with ``rmmod lisa && modprobe (..)``.
      As loading the module in ways different than through Lisa is not officially supported, any such
      setup is the user's responsibility.

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
