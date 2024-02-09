**Scope**

Ensure that the bug is not due to something installed from ``external/``
folder.  See ``external/subtrees.conf`` for the appropriate bug report location 

Then give a rough description of the environment you work with and your
workflow (e.g. from a jupyterlab notebook, an automated CI etc). This will
allow us to appropriately advise you on how to proceed.

**Describe the bug**

A clear and concise description of what the bug is.

**To Reproduce**

Steps to reproduce the behavior:
* how LISA was installed:
  * source of the LISA code: pip (PyPI), github (what sha1 ?)
  * what environment: on your host (what distro/version ?), in a VM, etc
* what part of LISA is being used (synthetic tests, WA-related features, etc.)
* what kind of target is being used (linux, android, etc.)
* what code is being run if that is a custom script using LISA APIs

**Expected behavior/what was the purpose of the experiment**

A clear and concise description of what you expected to happen.
Giving the overall purpose of the script you are trying to get to work can help
us identify which part of the API you may benefit from much more quickly.

**Logs of the error**

info and debug can be found under the artifacts as INFO.log and DEBUG.log.
The header of the log is useful as it gives plenty of information on python
version and similar things.
*Review the log and strip out any credential that you are not comfortable
sharing before posting*

**Additional context**

Add any other context about the problem here.
