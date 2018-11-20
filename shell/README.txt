
All LISA Shell commands start with "lisa-", thus using shell completion it is
easy to get a list of all the available commands.

Here is a list of the main ones, with a short description.
For a longer description type "lisa-<command> help"

.:: Generic commands
--------------------

lisa-help    - Print this help, or command specific helps
lisa-version - Dump info on the LISA in use
lisa-buildroot-create-rootfs   - Create a buildroot based rootfs to be used as userland for testing

.:: Maintenance commands
------------------------

lisa-update          - Update submodules and LISA notebooks/tests
lisa-install         - Remove the previous venv and do a fresh ven install
lisa-activate-venv   - Activate the LISA venv, and create it if it does not exist
lisa-deactivate-venv - Deactivate the LISA venv, and create it if it does not exist

.:: Notebooks commands
----------------------

lisa-ipython - Start/Stop the IPython Notebook server

.:: Results analysis and Documentation
--------------------------------------

lisa-report  - Pretty format results of last test

.:: Test commands
--------------------------------------

lisa-test    - Run tests and assert behaviours

