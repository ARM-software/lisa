***************
LISA self-tests
***************

Introduction
============

To ensure everything behaves as expected at all times, LISA comes with some
self-tests, which is a mix of unit and behavioural tests.

From the root of LISA, you can run those tests like so:

>>> python3 -m nose
>>> # You can also target specific test modules
>>> python3 -m nose tests/test_test_bundle.py
>>> # Or even specific test classes
>>> python3 -m nose tests/test_test_bundle.py:BundleCheck

Writing self-tests
==================

You should strive to validate as much of your code as possible through
self-tests. It's a nice way to showcase that your code works, and also how it
works. On top of that, it makes sure that later changes won't break it.

It's possible to write tests that require a live target - see
:meth:`~tests.utils.create_local_target`. However, as these tests
are meant to be run by Travis as part of our pull-request validation, they have
to be designed to work on a target with limited privilege.

:class:`unittest.TestCase`

Utilities
=========

.. automodule:: tests.utils
   :members:

Implemented tests
=================

.. TODO:: Make those imports more generic

TestBundle
++++++++++

.. automodule:: tests.test_test_bundle
   :members:

EnergyModel
+++++++++++

.. automodule:: tests.test_energy_model
   :members:

Trace
+++++

.. automodule:: tests.test_trace
   :members:

wlgen rt-app
++++++++++++

.. automodule:: tests.test_wlgen_rtapp
   :members:
