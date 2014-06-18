#!/usr/bin/python
"""Small functions to help with plots"""

from matplotlib import pyplot as plt

GOLDEN_RATIO = 1.618034

def set_plot_size(width, height):
    """Set the plot size.

    This has to be called before calls to .plot()
    """
    if height is None:
        if width is None:
            height = 6
            width = 10
        else:
            height = width / GOLDEN_RATIO
    else:
        if width is None:
            width = height * GOLDEN_RATIO

    plt.figure(figsize=(width, height))

def normalize_title(title, opt_title):
    """
    Return a string with that contains the title and opt_title if it's not the empty string

    See test_normalize_title() for usage
    """
    if opt_title is not "":
        title = opt_title + " - " + title

    return title
