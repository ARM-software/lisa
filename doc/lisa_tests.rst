***************
LISA self-tests
***************

Introduction
============

To ensure everything behaves as expected at all times, LISA comes with some
self-tests, which is a mix of unit and behavioural tests.

From the root of LISA, you can run those tests like so:

>>> python3 -m "nose" lisa/tests/lisa
>>> # You can also target specific test modules
>>> python3 -m "nose" lisa/tests/lisa/test_bundle.py
>>> # Or even specific test classes
>>> python3 -m "nose" lisa/tests/lisa/test_bundle.py:BundleCheck

Writing self-tests
==================

You should strive to validate as much of your code as possible through
self-tests. It's a nice way to showcase that your code works, and also how it
works. On top of that, it makes sure that later changes won't break it.

It's possible to write tests that require a live target - see
:meth:`~lisa.tests.lisa.utils.create_local_testenv`. However, as these tests
are meant to be run by Travis as part of our pull-request validation, they have
to be designed to work on a target with limited privilege.

Utilities
=========

.. automodule:: lisa.tests.lisa.utils
   :members:

Implemented tests
=================

.. TODO:: Make those imports more generic

TestBundle
++++++++++

.. automodule:: lisa.tests.lisa.test_test_bundle
   :members:

EnergyModel
+++++++++++

.. automodule:: lisa.tests.lisa.test_energy_model
   :members:

wlgen rt-app
++++++++++++

.. automodule:: lisa.tests.lisa.test_wlgen_rtapp
   :members:
