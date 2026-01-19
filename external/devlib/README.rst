devlib
======

``devlib`` exposes an interface for interacting with and collecting
measurements from a variety of devices (such as mobile phones, tablets and
development boards) running a Linux-based operating system.


Installation
------------

::

        sudo -H pip install devlib


Dependencies
------------

``devlib`` should install all dependencies automatically, however if you run
into issues please ensure you are using that latest version of pip.

On some systems there may additional steps required to install the dependency
``paramiko`` please consult the `module documentation <http://www.paramiko.org/installing.html>`_
for more information.

Usage
-----

Please refer  to the "Overview" section of the `documentation <http://devlib.readthedocs.io/en/latest/>`_.


Documentation
-------------

You can view pre-built HTML documentation `here <http://devlib.readthedocs.io/en/latest/>`_.

Documentation in reStructuredText format may be found under ``doc/source``. It
can be compiled into cross-linked HTML using `Sphinx <http://sphinx-doc.org>`_.


Generate
~~~~~~~~

To satisfy the dependencies in an automated way, the documentation can be
generated inside a python virtual environment. Even in this case, ensure that
the ``dot`` tool from ``graphviz`` is available on your system.

Once a fresh virtual environment is created and activated in your shell,
navigate to the devlib's root folder and set it up as follows: ::

        pip install -r doc/requirements.txt
        pip install -e .


After that, the documentation can be generated with the following commands: ::

        cd doc
        make html


The generated HTML documentation will be available in the
``doc/_build/html`` folder.

License
-------

This package is distributed under `Apache v2.0 License <http://www.apache.org/licenses/LICENSE-2.0>`_.


Feedback, Contributions and Support
-----------------------------------

- Please use the GitLab Issue Tracker associated with this repository for
  feedback.
- ARM licensees may contact ARM directly via their partner managers.
- We welcome code contributions via GitLab Merge requests. Please try to
  stick to the style in the rest of the code for your contributions.

