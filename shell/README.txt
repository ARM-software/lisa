
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

lisa-install         - Remove the previous venv and do a fresh ven install
lisa-activate-venv   - Activate the LISA venv, and create it if it does not exist
lisa-deactivate-venv - Deactivate the LISA venv, and create it if it does not exist
lisa-update-subtrees - Update the subtrees by pulling their latest changes
lisa-log-subtree     - Git log on the squashed commits of the give subtree. All other 
                       options are passed to `git log`.

.:: Notebooks commands
----------------------

lisa-jupyter - Start/Stop the Jupyter Notebook server

.:: Test commands
--------------------------------------

lisa-test    - Run tests and assert behaviours.
               This is just a wrapper around exekall that selects all tests
               modules and use positional arguments as --select patterns.  Also
               the default configuration file ($LISA_HOME/target_conf.yml) will
               be used if available (but this can be extended with
               user-supplied --conf).

