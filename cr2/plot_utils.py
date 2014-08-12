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

def pre_plot_setup(width=None, height=None, ncols=1, nrows=1):
    """initialize a figure

    width and height are the height and width of each row of plots.
    For 1x1 plots, that's the height and width of the plot.  This
    function should be called before any calls to plot()

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

    height *= nrows

    _, axis = plt.subplots(ncols=ncols, nrows=nrows, figsize=(width, height))

    # Needed for multirow blots to not overlap with each other
    plt.tight_layout(h_pad=3.5)

    return axis

def post_plot_setup(ax, title="", xlabel=None, xlim="default", ylim="range"):
    """Set xlabel, title, xlim adn ylim of the plot

    This has to be called after calls to .plot().  The default ylim is
    to extend it by 10% because matplotlib default makes it hard
    values that are close to the margins

    """

    if xlabel is not None:
        plt.xlabel(xlabel)

    if title:
        ax.set_title(title)

    set_ylim(ax, ylim)
    set_xlim(ax, xlim)

def plot_temperature(runs, width=None, height=None, ylim="range"):
    """Plot temperatures

    runs is an array of Run() instances.  Extract the control_temp
    from the governor data and plot the temperatures reported by the
    thermal framework.  The governor doesn't track temperature when
    it's off, so the thermal framework trace is more reliable.

    """

    ax = pre_plot_setup(width, height)

    for run in runs:
        current_temp = run.thermal_governor.data_frame["current_temperature"]
        delta_temp = run.thermal_governor.data_frame["delta_temperature"]
        control_temp_series = (current_temp + delta_temp) / 1000
        run.thermal.plot_temperature(control_temperature=control_temp_series,
                                     ax=ax, legend_label=run.name)

    post_plot_setup(ax, title="Temperature", ylim=ylim)
    plt.legend(loc="best")

def plot_hist(data, title, bins, xlabel, xlim, ylim):
    """Plot a histogram"""

    mean = data.mean()
    std = data.std()
    title += " (mean = {:.2f}, std = {:.2f})".format(mean, std)

    ax = pre_plot_setup()
    data.hist(ax=ax, bins=bins)
    post_plot_setup(ax, title=title, xlabel=xlabel, xlim=xlim, ylim=ylim)

def plot_load(runs, map_label, width=None, height=None):
    """Make a multiplot of all the loads"""
    num_runs = len(runs)
    axis = pre_plot_setup(width=width, height=height, ncols=num_runs)

    if num_runs == 1:
        axis = [axis]

    for ax, run in zip(axis, runs):
        run.in_power.plot_load(map_label, ax=ax)

        title = normalize_title("Utilisation", run.name)
        post_plot_setup(ax, title=title)

def plot_allfreqs(runs, map_label, width=None, height=None):
    """Make a multicolumn plots of the allfreqs plots of each run"""
    num_runs = len(runs)
    axis = pre_plot_setup(width=width, height=height, nrows=len(map_label),
                          ncols=num_runs)

    if num_runs == 1:
        axis = [axis]
    else:
        axis = zip(*axis)

    for ax, run in zip(axis, runs):
        run.plot_allfreqs(map_label, ax=ax)
