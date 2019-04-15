*******************
Contributor's guide
*******************

Contribution rules
==================

See `this document <https://github.com/ARM-software/lisa/blob/next/CONTRIBUTING.md>`__.

Subtrees
========

are available as subtrees under ``$repo/external``.

Updating the subtrees
+++++++++++++++++++++

If you got a Pull Request merged in e.g. :mod:`devlib` and want to use some of
the features you introduced in LISA, you'll need to update the subtrees. There is
a handy LISA shell command available for that:

  >>> lisa-update-subtrees

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
self-tests, which is a mix of unit and behavioural tests.

From the root of LISA, you can run those tests like so:

>>> python3 -m nose
>>> # You can also target specific test modules
>>> python3 -m nose tests/test_test_bundle.py
>>> # Or even specific test classes
>>> python3 -m nose tests/test_test_bundle.py:BundleCheck

Writing self-tests
++++++++++++++++++

You should strive to validate as much of your code as possible through
self-tests. It's a nice way to showcase that your code works, and also how it
works. On top of that, it makes sure that later changes won't break it.

It's possible to write tests that require a live target - see
:meth:`~tests.utils.create_local_target`. However, as these tests
are meant to be run by Travis as part of our pull-request validation, they have
to be designed to work on a target with limited privilege.

API
+++

Utilities
---------

.. automodule:: tests.utils
   :members:

Implemented tests
-----------------

.. TODO:: Make those imports more generic

.. automodule:: tests.test_test_bundle
   :members:

.. automodule:: tests.test_energy_model
   :members:

.. automodule:: tests.test_trace
   :members:

.. automodule:: tests.test_wlgen_rtapp
   :members:
