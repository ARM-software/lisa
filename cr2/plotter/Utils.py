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
"""Utils module has generic utils that will be used across
objects
"""


def listify(to_select):
    """Utitlity function to handle both single and
    list inputs
    """

    if not isinstance(to_select, list):
        to_select = [to_select]

    return to_select


def normalize_list(val, lst):
    """Normalize a unitary list"""

    if len(lst) != 1:
        raise RuntimeError("Cannot Normalize a non-unitary list")

    return lst * val


def decolonize(val):
    """Remove the colon at the end of the word
    This will be used by the unique word of
    template class to sanitize attr accesses
    """

    return val.strip(":")
