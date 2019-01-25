*******************
Contributor's guide
*******************

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

Contribution rules
==================

See `this document <https://github.com/ARM-software/lisa/blob/next/CONTRIBUTING.md>`__.
