***************
LISA self-tests
***************

To ensure everything behaves as expected at all times, LISA comes with some
self-tests, which is a mix of unit and behavioural tests.

From the root of LISA, you can run those tests like so:

>>> python3 -m "nose" lisa/tests/lisa
>>> # You can also target specific test modules
>>> python3 -m "nose" lisa/tests/lisa/test_bundle.py
>>> # Or even specific test classes
>>> python3 -m "nose" lisa/tests/lisa/test_bundle.py:BundleCheck

.. TODO:: Make those imports more generic
.. automodule:: lisa.tests.lisa.test_bundle
   :members:
