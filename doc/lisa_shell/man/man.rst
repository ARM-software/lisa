**********
LISA shell
**********

Description
+++++++++++

Once you have all of the required dependencies installed, you can use the LISA
shell, which provides a convenient set of commands for easy access to many LISA
related functions, scripts and environment variables.

For more details, see
`<https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/>`_

Activation
++++++++++

In order to use the shell, source the script:

.. code-block:: sh

   source init_env

.. note::

   This is done automatically by vagrant, so you don't have to issue this
   command after doing a ``vagrant ssh``

.. tip:: Run ``man lisa`` to see an overview of the provided LISA commands.


Commands
++++++++

Most LISA Shell commands start with ``lisa-``, thus using shell completion it
is easy to get a list of all the available commands.

Here is the documentation of the simple ones, more complex scripts have
integrated ``--help`` documentation, a section in the online documentation, or
man pages.

Maintenance commands
--------------------


* ``lisa-help``             - Show this help
* ``lisa-install``          - Remove the previous venv and do a fresh ven install
* ``lisa-version``          - Dump info on the LISA in use
* ``lisa-activate-venv``    - Activate the LISA venv, and create it if it does not exist
* ``lisa-deactivate-venv``  - Deactivate the LISA venv, and create it if it does not exist
* ``lisa-update-subtrees``  - Update the subtrees by pulling their latest changes
* ``lisa-log-subtree``      - Git log on the squashed commits of the given
  subtree. All other options are passed to `git log`.
* ``lisa-doc-build``        - Build the documentation
* ``lisa-build-asset``      - Download and cross-compile the binary assets in `lisa/assets/binaries`


Notebooks commands
------------------

* ``lisa-jupyter`` - Start/Stop the Jupyter Notebook server.

   Usage: ``lisa-jupyter CMD [NETIF [PORT]]``

   .. list-table::
      :widths: auto
      :align: left

      * - `CMD`
        - `start` to start the jupyter notebook server, `stop` to stop it
          (default: `start`)
      * - `NETIF`
        - the network interface to start the server on (default: `lo`)
      * - `PORT`
        - the tcp port for the server (default: 8888)

* ``lisa-execute-notebook`` - Execute the given notebook as a script.

Test commands
-------------

* ``lisa-test`` - Run LISA synthetic tests.

   This is just a wrapper around ``exekall`` that selects all tests modules and
   use positional arguments as ``--select`` patterns. The default configuration
   file (``$LISA_CONF``) will be used if available. This can be extended with
   user-supplied ``--conf``. If multiple iterations are requested using ``-n``,
   the :class:`lisa.target.Target` instance will be reused across iterations,
   to avoid the overhead of setting up the target environment.

   Usage: ``lisa-test TEST_PATTERN ... [EXEKALL_OPTIONS ...]``

   Example: ``lisa-test 'OneSmallTask*' --list``


* ``lisa-wltest-series``    - See :ref:`wltest main documentation<wltest-doc>`
* ``exekall``               - See :ref:`exekall main documentation<exekall-doc>`
* ``bisector``              - See :ref:`bisector main documentation<bisector-doc>`

Buildroot commands
------------------

* ``lisa-buildroot-create-rootfs``- Create a buildroot based rootfs to be used
  as userland for testing
* ``lisa-buildroot-update-kernel-config`` - Update a kernel config to bake a
  buildroot initramfs into the kernel.

Misc commands
-------------

* ``lisa-plot`` - Generate various plots from a ``trace.dat`` file.
  See ``lisa-plot -h`` for available plots.

Environment variables
+++++++++++++++++++++

The following environment variables are available:

.. run-command::

  # Strip-out version-specific info, so we have a more stable output
  export LISA_VENV_PATH=".lisa-venv-<python version>"
  env-list.py --rst --filter-home

If an environment variable is defined prior to sourcing ``init_env``, it will
keep its value.
