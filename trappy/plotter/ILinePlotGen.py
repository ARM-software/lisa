#    Copyright 2015-2015 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This is helper module for :mod:`trappy.plotter.ILinePlot`
for adding HTML and javascript necessary for interactive
plotting. The Linear to 2-D co-ordination transformations
are done by using the functionality in
:mod:`trappy.plotter.PlotLayout`
"""

from trappy.plotter import AttrConf
import uuid
import json
import os
from trappy.plotter import IPythonConf


if not IPythonConf.check_ipython():
    raise ImportError("No Ipython Environment found")

from IPython.display import display, HTML


class ILinePlotGen(object):
    """
    :param cols: The number of columns to draw
    :type cols: int

    :param num_plots: The total number of plots
    :type num_plots: int

    The linear co-ordinate system :math:`[0, N_{plots}]` is
    mapped to a 2-D coordinate system with :math:`N_{rows}`
    and :math:`N_{cols}` such that:

    .. math::

        N_{rows} = \\frac{N_{cols}}{N_{plots}}
    """

    def _add_graph_cell(self, fig_name):
        """Add a HTML table cell to hold the plot"""

        width = int(self._attr["width"] / self._cols)
        div_js = """
            <script>
            var ilp_req = require.config( {

                paths: {
                    "dygraph-sync": '""" + IPythonConf.add_web_base("plotter_scripts/ILinePlot/synchronizer") + """',
                    "dygraph": '""" + IPythonConf.add_web_base("plotter_scripts/ILinePlot/dygraph-combined") + """',
                    "ILinePlot": '""" + IPythonConf.add_web_base("plotter_scripts/ILinePlot/ILinePlot") + """',
                    "underscore": '""" + IPythonConf.add_web_base("plotter_scripts/ILinePlot/underscore-min") + """',
                },

                shim: {
                    "dygraph-sync": ["dygraph"],
                    "ILinePlot": {

                        "deps": ["dygraph-sync", "dygraph", "underscore"],
                        "exports":  "ILinePlot"
                    }
                }
            });
                ilp_req(["require", "ILinePlot"], function() {
                ILinePlot.generate('""" + fig_name + """', '""" + IPythonConf.add_web_base("") + """');
            });
            </script>
        """

        cell = '<td style="border-style: hidden;"><div class="ilineplot" id="{0}" style="width: \
{1}px; height: {2}px;">{3}</div></td>'.format(fig_name,
                                           width,
                                           self._attr["height"], div_js)

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

    def _end_table(self):
        """Add the closing tag for the HTML table"""

        self._html.append("</table>")

    def _generate_fig_name(self):
        """Generate a unique figure name"""

        fig_name = "fig_" + uuid.uuid4().hex
        self._fig_map[self._fig_index] = fig_name
        self._fig_index += 1
        return fig_name

    def _init_html(self):
        """Initialize HTML code for the plots"""

        width = self._attr["width"]
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

        self._end_table()

    def __init__(self, cols, num_plots, **kwargs):

        self._cols = cols
        self._attr = kwargs
        self._html = []
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

        self._attr["width"] = AttrConf.HTML_WIDTH
        self._attr["height"] = AttrConf.HTML_HEIGHT
        self._init_html()

    def _check_add_scatter(self, fig_params):
        """Check if a scatter plot is needed
        and augment the fig_params accordingly"""

        if self._attr["scatter"]:
            fig_params["drawPoints"] = True
            fig_params["strokeWidth"] = 0.0
        else:
            fig_params["drawPoints"] = False
            fig_params["strokeWidth"] = AttrConf.LINE_WIDTH

        fig_params["pointSize"] = self._attr["point_size"]

    def add_plot(self, plot_num, data_frame, title=""):
        """Add a plot for the corresponding index

        :param plot_num: The linear index of the plot
        :type plot_num: int

        :param data_frame: The data for the plot
        :type data_frame: :mod:`pandas.DataFrame`

        :param title: The title for the plot
        :type title: str
        """

        fig_name = self._fig_map[plot_num]
        fig_params = {}
        fig_params["data"] = json.loads(data_frame.to_json())
        fig_params["name"] = fig_name
        fig_params["rangesel"] = False
        fig_params["logscale"] = False
        fig_params["title"] = title
        fig_params["step_plot"] = self._attr["step_plot"]
        fig_params["fill_graph"] = self._attr["fill"]

        self._check_add_scatter(fig_params)

        if "group" in self._attr:
            fig_params["syncGroup"] = self._attr["group"]
            if "sync_zoom" in self._attr:
                fig_params["syncZoom"] = self._attr["sync_zoom"]
            else:
                fig_params["syncZoom"] = AttrConf.DEFAULT_SYNC_ZOOM

        if "ylim" in self._attr:
            fig_params["valueRange"] = self._attr["ylim"]

        json_file = os.path.join(IPythonConf.get_data_path(), fig_name + ".json")
        fh = open(json_file, "w")
        json.dump(fig_params, fh)
        fh.close()

    def finish(self):
        """Called when the Plotting is finished"""

        figs = []

        for fig_idx in self._fig_map.keys():
            figs.append(self._fig_map[fig_idx])

        display(HTML(self.html()))

    def html(self):
        """Return the raw HTML text"""

        return "\n".join(self._html)
