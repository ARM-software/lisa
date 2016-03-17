===========
Conventions
===========

Interface Definitions
=====================

Throughout this documentation a number of stubbed-out class definitions will be
presented showing an interface defined by a base class that needs to be
implemented by the deriving classes. The following conventions will be used when
presenting such an interface:

   - Methods shown raising :class:`NotImplementedError` are abstract and *must*
     be overridden by subclasses.
   - Methods with ``pass`` in their body *may* be (but do not need to be)  overridden 
     by subclasses. If not overridden, these methods will default to the base
     class implementation, which may or may not be a no-op (the ``pass`` in the
     interface specification does not necessarily mean that the method does not have an 
     actual implementation in the base class).

     .. note:: If you *do* override these methods you must remember to call the
               base class' version inside your implementation as well.

   - Attributes who's value is shown as ``None`` *must* be redefined by the
     subclasses with an appropriate value.
   - Attributes who's value is shown as something other than ``None`` (including
     empty strings/lists/dicts) *may* be (but do not need to be) overridden by 
     subclasses. If not overridden, they will default to the value shown.

Keep in mind that the above convention applies only when showing interface
definitions and may not apply elsewhere in the documentation. Also, in the
interest of clarity, only the relevant parts of the base class definitions will
be shown some members (such as internal methods) may be omitted.


Code Snippets
=============

Code snippets provided are intended to be valid Python code, and to be complete.
However, for the sake of clarity, in some cases only the relevant parts will be
shown with some details omitted (details that may necessary to validity of the code 
but not to understanding of the concept being illustrated). In such cases, a
commented ellipsis will be used to indicate that parts of the code have been
dropped. E.g.  ::

        # ...

        def update_result(self, context):
           # ...
           context.result.add_metric('energy', 23.6, 'Joules', lower_is_better=True)

        # ...


Core Class Names
================

When core classes are referenced throughout the documentation, usually their
fully-qualified names are given e.g. :class:`wlauto.core.workload.Workload`.
This is done so that Sphinx_ can resolve them and provide a link. While
implementing plugins, however, you should *not* be importing anything
directly form under :mod:`wlauto.core`. Instead, classes you are meant to
instantiate or subclass have been aliased in the root :mod:`wlauto` package,
and should be imported from there, e.g. ::

        from wlauto import Workload

All examples given in the documentation follow this convention. Please note that
this only applies to the :mod:`wlauto.core` subpackage; all other classes
should be imported for their corresponding subpackages.

.. _Sphinx: http://sphinx-doc.org/


