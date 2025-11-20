#! /usr/bin/env python3

def pytest_configure():
    import os
    os.environ['LISA_EXTRA_ASSERTS'] = '1'
    import lisa.utils
    assert lisa.utils._EXTRA_ASSERTS
