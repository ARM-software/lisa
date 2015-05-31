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
# File:        ILinePlotGen.py
# ----------------------------------------------------------------
# $
#
"""This module is reponsible for creating a layout
of plots as a 2D axes and handling corener cases
and deleting empty plots
"""

from cr2.plotter import AttrConf
import uuid
import json
import os
from IPython.display import display, HTML

if not AttrConf.PLOTTER_IPYTHON:
    raise ImportError("No Ipython Environment found")


class ILinePlotGen(object):

    """Cols is the number of columns to draw
       rows are calculated as 1D - 2D transformation
       the same transformation is used to index the
       axes array
    """

    def _add_lib(self):
        """Add Library String"""
        lib_str = '<script src="{0}"></script>'.format(
            AttrConf.DYGRAPH_LIB_URL)
        self._html.append(lib_str)

    def _add_graph_cell(self, fig_name):
        """Add a HTML table cell to hold the plot"""

        width = int(self._attr["width"] / self._cols)

        cell = '<td style="border-style: hidden;"><div class="ilineplot" id="{0}" style="width: \
{1}px; height: {2}px;"></div></td>'.format(fig_name,
                                           width,
                                           self._attr["height"])

        self._html.append(cell)

    def _add_legend_cell(self, fig_name):
        """Add HTML table cell for the legend"""

        width = int(self._attr["width"] / self._cols)
        legend_div_name = fig_name + "_legend"
        cell = '<td style="border-style: hidden;"><div style="text-align:right; \
width: {0}px; height: auto;"; id="{1}"></div></td>'.format(width,
                                                           legend_div_name)

        self._html.append(cell)

    def _begin_row(self):
        """Add the opening tag for HTML row"""

        self._html.append("<tr>")

    def _end_row(self):
        """Add the closing tag for the HTML row"""

        self._html.append("</tr>")

    def _generate_fig_name(self):
        """Generate a unique figure name"""

        fig_name = "fig_" + uuid.uuid4().hex
        self._fig_map[self._fig_index] = fig_name
        self._fig_index += 1
        return fig_name

    def _init_html(self):
        """Initialize HTML code for the plots"""

        width = self._attr["width"]
        self._add_lib()
        table = '<table style="width: {0}px; border-style: hidden;">'.format(
            width)
        self._html.append(table)

        for _ in range(self._rows):
            self._begin_row()

            legend_figs = []
            for _ in range(self._cols):
                fig_name = self._generate_fig_name()
                legend_figs.append(fig_name)
                self._add_graph_cell(fig_name)

            self._end_row()
            self._begin_row()

            for l_fig in legend_figs:
                self._add_legend_cell(l_fig)

            self._end_row()
        display(HTML(self.html()))

    def __init__(self, cols, num_plots, step_plot=False):
        """
            Args:
                cols (int): Number of plots in a single line
                num_plots (int): Total Number of Plots
                step_plot (bool, default=False): Draw step-post
                    plots
        """

        self._cols = cols
        self._attr = {}
        self._html = []
        self._js = []
        self.num_plots = num_plots
        self._fig_map = {}
        self._fig_index = 0

        self._single_plot = False
        if self.num_plots == 0:
            raise RuntimeError("No plots for the given constraints")

        if self.num_plots < self._cols:
            self._cols = self.num_plots
        self._rows = (self.num_plots / self._cols)

        if self.num_plots % self._cols != 0:
            self._rows += 1

        self._attr = {}
        self._attr["width"] = AttrConf.HTML_WIDTH
        self._attr["height"] = AttrConf.HTML_HEIGHT
        self._attr["step_plot"] = step_plot
        self._init_html()
        self._init_js()

    def add_plot(self, plot_num, data_frame, title=""):
        """Add a plot to for a corresponding index"""

        fig_name = self._fig_map[plot_num]
        fig_params = {}
        fig_params["data"] = json.loads(data_frame.to_json())
        fig_params["name"] = fig_name
        fig_params["rangesel"] = False
        fig_params["logscale"] = False
        fig_params["title"] = title
        fig_params["step_plot"] = self._attr["step_plot"]

        json_file = os.path.join(AttrConf.PLOTTER_STATIC_DATA_DIR, fig_name + ".json")
        fh = open(json_file, "w")
        json.dump(fig_params, fh)
        fh.close()

    def finish(self):
        """Called when the Plotting is finished"""

        figs = []

        for fig_idx in self._fig_map.keys():
            figs.append(self._fig_map[fig_idx])

        display(HTML(self.js()))

    def _init_js(self):
        """Initialize Javascript for the plots"""

        base_dir = os.path.dirname(os.path.realpath(__file__))
        js_file = os.path.join(base_dir, "js/ILinePlot.js")

        js_fh = open(js_file, 'r')
        self._js = js_fh.readlines()
        js_fh.close()

    def js(self):
        """Return the raw js text"""

        return "\n".join(self._js)

    def html(self):
        """Return the raw HTML text"""

        return "\n".join(self._html)
