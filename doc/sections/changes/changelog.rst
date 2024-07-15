*********
Changelog
*********

.. exec::

    from lisa.utils import LISA_HOME
    from lisa._doc.helpers import make_changelog

    repo = LISA_HOME
    changelog = make_changelog(
        repo=LISA_HOME,
    )
    print(changelog)
