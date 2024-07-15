.. _api-stability-page:

**************
APIs stability
**************

APIs inside LISA are split between private and public ones:

* Public APIs can be expected to stay stable, or undergo a deprecation cycle
  where they will trigger an :exc:`DeprecationWarning` and be documented as
  such before being removed. Exceptions to that rule are documented explicitly
  as such.

* Private APIs can be changed at all points.

Public APIs consist of classes and functions with names not starting with an
underscore, defined in modules with a name not starting with an underscore (or
any of its parent modules or containing class).

Everything else is private.

.. note:: User subclassing is usually more at risk of breakage than other uses
    of the APIs. Behaviors are usually not restricted to a single method, which
    means the subclass would have to override multiple of them to preserve
    important API laws. This is unfortunately not future-proof, as new versions
    can add new methods that would also require being overridden and kept in
    sync. If for some reason subclassing is required, please get in touch in the
    `GitLab issue tracker <https://gitlab.arm.com/tooling/lisa/-/issues>`_
    before relying on that for production.

.. note:: Instance attributes are considered public following the same
    convention as functions and classes. Only reading from them is expected in
    user code though, any attempt to modify or delete them is outside of the
    bounds of what the public API exposes (unless stated explicitly otherwise).
    This means that a minor version change could swap an instance attribute for
    a read-only property. It also means that any problem following the
    modification of an attribute by a user will not be considered as a bug.
