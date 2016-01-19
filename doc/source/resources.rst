.. _resources:

Dynamic Resource Resolution
===========================

Introduced in version 2.1.3.

The idea is to decouple resource identification from resource discovery.
Workloads/instruments/devices/etc state *what* resources they need, and not
*where* to look for them -- this instead is left to the resource resolver that
is now part of the execution context. The actual discovery of resources is
performed by resource getters that are registered with the resolver.

A resource type is defined by a subclass of
:class:`wlauto.core.resource.Resource`. An instance of this class describes a
resource that is to be obtained. At minimum, a ``Resource`` instance has an
owner (which is typically the object that is looking for the resource), but
specific resource types may define other parameters that describe an instance of
that resource (such as file names, URLs, etc).

An object looking for a resource invokes a resource resolver with an instance of
``Resource`` describing the resource it is after. The resolver goes through the
getters registered for that resource type in priority order attempting to obtain
the resource; once the resource is obtained, it is returned to the calling
object. If none of the registered getters could find the resource, ``None`` is
returned instead.

The most common kind of object looking for resources is a ``Workload``, and
since v2.1.3, ``Workload`` class defines
:py:meth:`wlauto.core.workload.Workload.init_resources` method that may be
overridden by subclasses to perform resource resolution. For example, a workload
looking for an APK file would do so like this::

    from wlauto import Workload
    from wlauto.common.resources import ApkFile

    class AndroidBenchmark(Workload):

        # ...

        def init_resources(self, context):
                self.apk_file = context.resource.get(ApkFile(self))

        # ...


Currently available resource types are defined in :py:mod:`wlauto.common.resources`.
