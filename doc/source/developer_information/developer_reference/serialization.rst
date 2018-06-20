.. _serialization:

Serialization
=============

Overview of Serialization
-------------------------

WA employs a serialization mechanism in order to store some of its internal
structures inside the output directory. Serialization is performed in two
stages:

1. A serializable object is converted into a POD (Plain Old Data) structure
   consisting of primitive Python types, and a few additional types (see
   :ref:`wa-pods` below).
2. The POD structure is serialized into a particular format by a generic
   parser for that format. Currently, `yaml` and `json` are supported.

Deserialization works in reverse order -- first the serialized text is parsed
into a POD, which is then converted to the appropriate object.


Implementing Serializable Objects
---------------------------------

In order to be considered serializable, an object must either be a POD, or it
must implement the ``to_pod()`` method and ``from_pod`` static/class method,
which will perform the conversion to/form pod.

As an example, below as a (somewhat trimmed) implementation of the ``Event``
class:

.. code-block:: python

    class Event(object):

        @staticmethod
        def from_pod(pod):
            instance = Event(pod['message'])
            instance.timestamp = pod['timestamp']
            return instance

        def __init__(self, message):
            self.timestamp = datetime.utcnow()
            self.message = message

        def to_pod(self):
            return dict(
                timestamp=self.timestamp,
                message=self.message,
            )


Serialization API
-----------------

.. function:: read_pod(source, fmt=None)
.. function:: write_pod(pod, dest, fmt=None)

    These read and write PODs from a file. The format will be inferred, if
    possible, from the extension of the file, or it may be specified explicitly
    with ``fmt``. ``source`` and ``dest`` can be either strings, in which case
    they will be interpreted as paths, or they can be file-like objects.

.. function:: is_pod(obj)

    Returns ``True`` if ``obj`` is a POD, and ``False`` otherwise.

.. function:: dump(o, wfh, fmt='json', \*args, \*\*kwargs)
.. function:: load(s, fmt='json', \*args, \*\*kwargs)

    These implment an altenative serialization interface, which matches the
    interface exposed by the parsers for the supported formats.


.. _wa-pods:

WA POD Types
------------

POD types are types that can be handled by a serializer directly, without a need
for any additional information. These consist of the build-in python types ::

    list
    tuple
    dict
    set
    str
    unicode
    int
    float
    bool

...the standard library types ::

    OrderedDict
    datetime

...and the WA-defined types ::

    regex_type
    none_type
    level
    cpu_mask

Any structure consisting entirely of these types is a POD and can be serialized
and then deserialized without losing information. It is important to note that
only these specific types are considered POD, their subclasses are *not*.

.. note:: ``dict``\ s get deserialized as ``OrderedDict``\ s.


Serialization Formats
---------------------

WA utilizes two serialization formats: YAML and JSON. YAML is used for files
intended to be primarily written and/or read by humans; JSON is used for files
intended to be primarily written and/or read by WA and other programs.

The parsers and serializers for these formats used by WA have been modified to
handle additional types (e.g. regular expressions) that are typically not
supported by the formats. This was done in such a way that the resulting files
are still valid and can be parsed by any parser for that format.
