# $Copyright:
# ----------------------------------------------------------------
# This confidential and proprietary software may be used only as
# authorised by a licensing agreement from ARM Limited
#  (C) COPYRIGHT 2015 ARM Limited
#       ALL RIGHTS RESERVED
# The entire notice above must be reproduced on all authorised
# copies and copies may only be made to the extent permitted
# by a licensing agreement from ARM Limited.
# ----------------------------------------------------------------
# File:        Utils.py
# ----------------------------------------------------------------
# $
#
"""Utility functions for sheye"""

import cr2

def init_run(trace):
    """Initialize the Run Object"""

    if isinstance(trace, basestring):
        return cr2.Run(trace)

    elif isinstance(trace, cr2.Run):
        return trace

    raise ValueError("Invalid trace Object")
