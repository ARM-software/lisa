**********
Versioning
**********

LISA releases on :ref:`PyPI<setup-pypi>` are done following semantic versioning
as defined in https://semver.org/. As pointed by :ref:`api-stability-page`, classes are
split on the following axes for the purpose of semver tracking:

* A set of methods and attributes in general: Adding a method entails a minor
  version bump, even though it can technically cause a breaking change in a
  user subclass that happened to use the same name.

* Inheritance tree: the MRO of a class is not considered as part of the stable
  public API and can therefore change at any point. Classes named ``*Base``
  can usually be relied on for ``issubclass()`` and ``isinstance()`` but that
  is not a hard rule. The reason behind that is that even adding a class to
  the hierarchy can break existing uses of ``isinstance()`` so there is
  essentially no way of making any change to the inheritance tree that is not
  a breaking change.
