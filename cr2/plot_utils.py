#!/usr/bin/python
"""Small functions to help with plots"""

from matplotlib import pyplot as plt

GOLDEN_RATIO = 1.618034

def normalize_title(title, opt_title):
    """
    Return a string with that contains the title and opt_title if it's not the empty string

    See test_normalize_title() for usage
    """
    if opt_title is not "":
        title = opt_title + " - " + title

    return title

def set_lim(lim, get_lim, set_lim):
    """Set x or y limitis of the plot

    lim can be a tuple containing the limits or the string "default"
    or "range".  "default" does nothing and uses matplotlib default.
    "range" extends the current margin by 10%.  This is useful since
    the default xlim and ylim of the plots sometimes make it harder to
    see data that is just in the margin.

    """
    if lim == "default":
        return

    if lim == "range":
        cur_lim = get_lim()
        lim = (cur_lim[0] - 0.1 * (cur_lim[1] - cur_lim[0]),
               cur_lim[1] + 0.1 * (cur_lim[1] - cur_lim[0]))

    set_lim(lim[0], lim[1])

def set_xlim(ax, xlim):
    """Set the xlim of the plot

    See set_lim() for the details
    """
    set_lim(xlim, ax.get_xlim, ax.set_xlim)

def set_ylim(ax, ylim):
    """Set the ylim of the plot

    See set_lim() for the details
    """
    set_lim(ylim, ax.get_ylim, ax.set_ylim)

def pre_plot_setup(width=None, height=None):
    """initialize a figure

    width and height are numbers.  This function should be called
    before any calls to plot()

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

    _, ax = plt.subplots(figsize=(width, height))

    return ax

def post_plot_setup(ax, title="", xlabel=None, xlim="default", ylim="range"):
    """Set xlabel, title, xlim adn ylim of the plot

    This has to be called after calls to .plot().  The default ylim is
    to extend it by 10% because matplotlib default makes it hard
    values that are close to the margins

    """

    if xlabel is not None:
        plt.xlabel(xlabel)

    if title:
        plt.title(title)

    set_ylim(ax, ylim)
    set_xlim(ax, xlim)

def plot_allfreqs(in_power, out_power, map_label, title="", width=None, height=None):
    """Do allfreqs plots similar to those of CompareRuns"""
    import power

    all_freqs = power.get_all_freqs_data(in_power, out_power, map_label)

    for label, dfr in all_freqs.iteritems():
        this_title = normalize_title("allfreqs " + label, title)

        ax = pre_plot_setup(width=width, height=height)
        dfr.plot(ax=ax)
        post_plot_setup(ax, title=this_title)

def __plot_power_hists(power_inst, map_label, what, title):
    """Helper function for plot_power_hists

    power_obj is either an InPower() or OutPower() instance.  what is
    a string: "in" or "out"

    """
    freqs = power_inst.get_all_freqs(map_label)
    for actor in freqs:
        mean = freqs[actor].mean()
        std = freqs[actor].std()
        this_title = "freq_{} {} (mean = {:.2f}, std = {:.2f})".format(what, actor,
                                                               mean, std)
        this_title = normalize_title(this_title, title)
        xlim = (0, freqs[actor].max())

        ax = pre_plot_setup()
        freqs[actor].hist(ax=ax, bins=20)
        post_plot_setup(ax, title=this_title, xlabel="Frequency (KHz)",
                        xlim=xlim, ylim="default")

def plot_power_hists(in_power, out_power, map_label, title=""):
    """Plot histograms for each actor input and output power"""

    __plot_power_hists(out_power, map_label, "out", title)
    __plot_power_hists(in_power, map_label, "in", title)
