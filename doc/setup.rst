.. _setup-page:

*****
Setup
*****

Installation
============

Native installation
+++++++++++++++++++

For now, you will need to clone LISA from `github <https://github.com/ARM-software/lisa>`_ ,
and then issue these commands::

  git clone https://github.com/ARM-software/lisa.git
  cd lisa
  # A few packages need to be installed, like python3 or kernelshark. Python
  # modules will be installed in a venv at the next step, without touching
  # any system-wide install location.
  sudo ./install_base_ubuntu.sh
  # On the first run, it will take care of creating a Python venv and populating it
  source init_env

In case the venv becomes unusable for some reason, the ``lisa-install``
shell command available after sourcing ``init_env`` will allow to create a new
clean venv from scratch.

Additional Python packages
--------------------------

``lisa-install`` will also install the content of
``$LISA_HOME/custom_requirements.txt`` if the file exists. That allows
re-installing a custom set of packages automatically when the venv needs to
regenerated.

Without automatic ``venv``
--------------------------

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
++++++++++++++++++++++++++++++++++

LISA provides a Vagrant recipe which automates the generation of a
VirtualBox based virtual machine pre-configured to run LISA. To generate and
use such a virtual machine you need:

- `VirtualBox <https://www.virtualbox.org/wiki/Downloads>`__
- `Vagrant <https://www.vagrantup.com/downloads.html>`__

Once these two components are available on your machine, issue these commands::

  git clone https://github.com/ARM-software/lisa.git
  cd lisa
  vagrant up

This last command builds and executes the VM according to the description provided
by the Vagrant file available in the root folder of the LISA source tree.

Once the VM installation is complete, you can access that VM with::

  vagrant ssh

Updating
========

Over time, we might change/add some dependencies to LISA. As such, if you
update your LISA repository, you should make sure your locally-installed
packages still match those dependencies. Sourcing ``init_env`` from a
new shell should suffice, which will hint the user if running ``lisa-install``
again is needed.

.. tip::

  A git **post-checkout** hook is provided in ``tools/post-checkout``. It will
  check that no ``setup.py`` file have been updated since last time
  ``lisa-install`` was executed. If a modification is detected, it will ask the
  user to run ``lisa-install`` again, since a dependency might have been added,
  or a version requirement might have been updated.

External dependencies
=====================

Kernel modules
++++++++++++++

The following modules are required to run Lisa tests against some kernels.

sched_tp
--------

From Linux v5.3, sched_load_cfs_rq and sched_load_se tracepoints are present in
mainline as bare tracepoints without any events in tracefs associated with
them.

To help expose these tracepoints (and any additional one we might require in
the future) as trace events, an external module is required and is provided
under the name of sched_tp in $LISA_HOME/tools/kmodules/sched_tp

Building a module
+++++++++++++++++

The process is standard Linux external module build step. Helper scripts are
provides too.

Build
-----

.. code-block:: sh

  $LISA_HOME/tools/kmodules/build_module path/to/kenrel path/to/kmodule [path/to/install/modules]

This will build the module against the provided kernel tree and install it in
``path/to/install/module`` if provided otherwise install it in
``$LISA_HOME/tools/kmodules``.

Clean
-----

.. code-block:: sh

  $LISA_HOME/tools/kmodules/clean_module path/to/kenrel path/to/kmodule

Highly recommended to clean when switching kernel trees to avoid unintentional
breakage for using stale binaries.

Pushing the module into the target
----------------------------------

You need to push the module into your rootfs either by installing it directly
there or use commands like ``scp`` to copy it into your device.

.. code-block:: sh

  scp -r $LISA_HOME/tools/kmoudles/lib username@ip:/

Loading the module
------------------

On the target run:

.. code-block:: sh

  modprobe sched_tp

Integrating the module in your kernel tree
++++++++++++++++++++++++++++++++++++++++++

If you're rebuilding your kernel tree anyway, it might be easier to integrate
the module into your kernel tree as a built-in module so that it's always
present.

Integrate using provided patch
------------------------------

.. code-block:: sh

  cd path/to/kernel && git am path/to/patch
