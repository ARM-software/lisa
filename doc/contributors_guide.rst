*******************
Contributor's guide
*******************

First of all, if you’re reading this, thanks for thinking about
contributing! This project is maintained by us Arm folks, but we welcome
contributions from anyone.

.. _submit-merge-request:

How to submit a merge request
=============================

Submitting a merge request requires forking the LISA repository in a similar
fashion to the typical GitHub workflow:

1. Create a GitLab account on https://gitlab.arm.com/tooling/lisa/. You can
   use an existing GitHub login if you want.

2. Fork the repository. This requires the fork permission on the account,
   which can be obtained by following:
   https://gitlab.arm.com/documentation/contributions

3. Push your branch to your fork (this might require setting up your SSH
   public key in your profile just like on GitHub).

4. Open the merge request. ⚠️ If your fork is a private fork, GitLab will
   default to opening a merge request against your own fork, and no-one
   will ever know of your contribution. When opening the MR, there is a
   "Change branches" link next to "From XXX into main". Click this link and
   select "tooling/lisa" in the "Target branch" project drop down. ⚠️

Merge requests that are primarily constituted of style reformatting will be
closed without comment unless the matter was discussed previously with the
maintainer. Note that such discussion will be expected to be carried with
arguments. Stating opinions or arguments of authority and such will lead to the
end of the discussion.

How to reach us
===============

If you’re hitting an error/bug and need help, it’s best to raise an
issue on `GitLab <https://gitlab.arm.com/tooling/lisa/-/issues>`__.

Coding style
============

As a rule of thumb, the code you write should follow the
`PEP-8 <https://www.python.org/dev/peps/pep-0008/>`__.

We strongly recommend using a code checker such as
`pylint <https://www.pylint.org/>`__, as it tracks unused imports/variables,
informs you when you can simplify a statement using Python features, and overall
just helps you write better code. However, we don't enforce any linter in merged
code.

Documentation
=============

Docstring style
+++++++++++++++

Docstring documentation should follow the ReST/Sphinx style. Classes,
class attributes and public methods must be documented. If deemed
necessary, private methods can be documented as well.

All in all, it should look like this:

.. code:: python

   def foo(a, b):
       """
       A one liner description

       :param a: A description for param a
       :type a: int

       :param b: A description for param b
       :type b: str

       Whatever extra description you might over as many lines as you need
       (but be reasonable)
       """
       pass

.. note:: LISA does not use type annotations as they have only been introduced
    recently and would currently conflict with exekall use (that is a solvable
    problem but has not been worked on yet). Parameters types must therefore be
    documented using the ``:type the_param: the_type`` in the function
    docstring.


References to classes should be made using ``:class:`path.to.TheClass```, same
goes for methods (``:meth:`...```) and functions (``:func:`...```).

Examples on how to use the API can sometimes be useful. They should be
introduced by ``**Example**::`` and located:

* In the module docstring if they involve multiple classes or functions from
  the module.
* In the class docstring if they involve multiple methods of the class.
* In the method/function otherwise.

How to build
++++++++++++

- Install ``doc`` optional dependencies of ``lisa`` package (``lisa-install``
  does that by default)
- Run:

  .. code:: shell

    lisa-doc-build

- Find the HTML in ``doc/_build/html``

Commits
=======

As for the shape of the commit, nothing out of the ordinary: just follow
the good old 50/72 rule (it’s okay if you bite off a few extra chars).

The header should highlight the impacted files/classes. The ‘lisa’
prefix can be omitted - for instance, if you’re modifying the
``lisa/wlgen/rta.py`` file, we’d expect a header of the shape
``lisa.wlgen.rta: ...``.

It should also contain a ``FIX``, ``FEATURE`` or ``BREAKING CHANGE`` tag that
will be used to generate the changelog, such as:

.. code-block:: text

  lisa.foo.bar: Fix some foobar

  FIX

  This fix fixes fixable fixtures by affixing an postfix operator.


When in doubt, have a look at the git log.

Subtrees
========

are available as subtrees under ``$repo/external``.

Updating the subtrees
+++++++++++++++++++++

If you got a Pull Request merged in e.g. :mod:`devlib` and want to use some of
the features you introduced in LISA, you'll need to update the subtrees. There is
a handy LISA shell command available for that: ``lisa-update-subtrees``.

This will update every subtree in the repository with the right incantation, and
the result can be pushed straight away to LISA as a Pull Request (or included in
a broader Pull Request).

Submitting your subtree changes
+++++++++++++++++++++++++++++++

Our changes to subtrees are often developped conjointly with LISA, so we write our
modifications directly in the subtrees. You can commit these changes in the LISA
repository, then shape those modifications into a git history ready to be pushed
using ``git subtree split``. Assuming you want to split a devlib change and have
a devlib remote set up in your repository, you'd have to issue the following::

  # Ensure refs are up to date
  git fetch devlib
  # Do the split
  git subtree split --prefix=external/devlib -b my-devlib-feature

This will give you a ``my-devlib-feature`` branch ready to be pushed. To make
things easier, we recommend setting up a remote to your devlib fork::

  git remote add devlib-me git@github.com:me/devlib.git

You can then push this branch to your devlib fork like so::

  git push -u devlib-me my-devlib-feature

Validating your changes
=======================

To ensure everything behaves as expected at all times, LISA comes with some
self-tests in ``tests/`` folder, which is a mix of unit and behavioural tests.

From the root of LISA, you can run those tests like so:

.. code-block:: sh

    python3 -m pytest
    # You can also target specific test modules
    python3 -m pytest tests/test_test_bundle.py
    # Or even specific test classes
    python3 -m pytest tests/test_test_bundle.py::BundleCheck
    # Or even specific test method
    python3 -m pytest tests/test_test_bundle.py::BundleCheck::test_init

Writing self-tests
++++++++++++++++++

You should strive to validate as much of your code as possible through
self-tests. It's a nice way to showcase that your code works, and also how it
works. On top of that, it makes sure that later changes won't break it.

It's possible to write tests that require a live target - see
``create_local_target()``. However, as these tests are meant to be run by the
CI as part of our pull-request validation, they have to be designed to work on
a target with limited privilege.


Updating binary tools
=====================

LISA comes with a number of prebuilt static binaries in
``lisa/_assets/binaries``. They are all built according to recipes in
``tools/recipes/``, and can be re-built and installed using e.g.:
``lisa-build-asset trace-cmd``. See ``lisa-build-asset --help`` for more
options.
