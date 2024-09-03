***************
Release Process
***************

Making a new release involves the following steps:

1. Update ``version_tuple`` in :mod:`lisa.version`.

2. Ensure LISA as a whole refers to relevant versions of:

   * Alpine Linux in :mod:`lisa._kmod`
   * Ubuntu in ``Vagrantfile``
   * Binary dependencies in :mod:`lisa._assets`
   * Android SDK installed by ``install_base.sh``
   * Java version used by Android SDK in ``install_base.sh``

3. Ensure LISA can work with currently published version of devlib.

4. Create a ``vX.Y.Z`` tag.

5. Make the Python wheel. See ``tools/make-release.sh`` for some
   indications on that part.

6. Install that wheel in a _fresh_ :ref:`Vagrant VM<setup-vagrant>`. Ensure
   that the VM is reinstalled from scratch and that the vagrant box in use is
   up to date.

7. Run ``tools/tests.sh`` in the VM and ensure no deprecated item scheduled
   for removal in the new version is still present in the sources (should
   result in import-time exceptions).

8. Ensure all CIs in use are happy.

9. Push the ``vX.Y.Z`` tag in the main repo

10. Update the ``release`` branch to be at the same commit as the ``vX.Y.Z`` tag.

11. Upload the wheel on PyPI.
