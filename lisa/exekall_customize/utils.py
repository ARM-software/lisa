#! /usr/bin/env python3

"""
This module hosts all the code that needs to be used by other parts of the lisa
package, without importing exekall. That way, we keep lisa independent from
exekall and at the same time, we allow integration.
"""

class IDHidden:
    pass

