#    Copyright 2015-2017 ARM Limited
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
from collections import OrderedDict
import json
import os
from trappy.plotter import IPythonConf
from trappy.plotter.ColorMap import to_dygraph_colors


if not IPythonConf.check_ipython():
    raise ImportError("No Ipython Environment found")

from IPython.display import display, HTML

def df_to_dygraph(data_frame):
    """Helper function to convert a :mod:`pandas.DataFrame` to
    dygraph data

    :param data_frame: The DataFrame to be converted
    :type data_frame: :mod:`pandas.DataFrame`
    """

    values = data_frame.values.tolist()
    data = [[x] for x in data_frame.index.tolist()]

    for idx, (_, val) in enumerate(zip(data, values)):
        data[idx] += val

    return {
        "data": data,
        "labels": ["index"] + data_frame.columns.tolist(),
    }

class ILinePlotGen(object):
    """
    :param num_plots: The total number of plots
    :type num_plots: int

    The linear co-ordinate system :math:`[0, N_{plots}]` is
    mapped to a 2-D coordinate system with :math:`N_{rows}`
    and :math:`N_{cols}` such that:

    .. math::

        N_{rows} = \\frac{N_{cols}}{N_{plots}}
    """

    def _add_graph_cell(self, fig_name, color_map):
        """Add a HTML table cell to hold the plot"""

        colors_opt_arg = ", " + to_dygraph_colors(color_map) if color_map else ""

        graph_js = ''
        lib_urls =  [IPythonConf.DYGRAPH_COMBINED_URL, IPythonConf.DYGRAPH_SYNC_URL,
                     IPythonConf.UNDERSCORE_URL]
        for url in lib_urls:
            graph_js += '<!-- TRAPPY_PUBLISH_SOURCE_LIB = "{}" -->\n'.format(url)

        graph_js += """
            <script>
            /* TRAPPY_PUBLISH_IMPORT = "plotter/js/ILinePlot.js" */
            /* TRAPPY_PUBLISH_REMOVE_START */
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
                /* TRAPPY_PUBLISH_REMOVE_STOP */
                ilp_req(["require", "ILinePlot"], function() { /* TRAPPY_PUBLISH_REMOVE_LINE */
                ILinePlot.generate(""" + fig_name + "_data" + colors_opt_arg + """);
            }); /* TRAPPY_PUBLISH_REMOVE_LINE */
            </script>
        """

        cell = '<td style="border-style: hidden;"><div class="ilineplot" id="{}"></div></td>'.format(fig_name)

        self._html.append(cell)
        self._js.append(graph_js)

    def _add_legend_cell(self, fig_name):
        """Add HTML table cell for the legend"""

        legend_div_name = fig_name + "_legend"
        cell = '<td style="border-style: hidden;"><div style="text-align:center" id="{}"></div></td>'.format(legend_div_name)

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

    def _init_html(self, color_map):
        """Initialize HTML code for the plots"""

        table = '<table style="border-style: hidden;">'
        self._html.append(table)
        if self._attr["title"]:
            cell = '<caption style="text-align:center; font: 24px sans-serif bold; color: black">{}</caption>'.format(self._attr["title"])
            self._html.append(cell)

        for _ in range(self._rows):
            self._begin_row()
            legend_figs = []
            for _ in range(self._attr["per_line"]):
                fig_name = self._generate_fig_name()
                legend_figs.append(fig_name)
                self._add_graph_cell(fig_name, color_map)

            self._end_row()
            self._begin_row()

            for l_fig in legend_figs:
                self._add_legend_cell(l_fig)

            self._end_row()

        self._end_table()

    def __init__(self, num_plots, **kwargs):

        self._attr = kwargs
        self._html = []
        self._js = []
        self._js_plot_data = []
        self.num_plots = num_plots
        self._fig_map = {}
        self._fig_index = 0

        self._single_plot = False
        if self.num_plots == 0:
            raise RuntimeError("No plots for the given constraints")

        if self.num_plots < self._attr["per_line"]:
            self._attr["per_line"] = self.num_plots
        self._rows = (self.num_plots / self._attr["per_line"])

        if self.num_plots % self._attr["per_line"] != 0:
            self._rows += 1

        self._attr["height"] = AttrConf.HTML_HEIGHT
        self._init_html(kwargs.pop("colors", None))

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

    def add_plot(self, plot_num, data_frame, title="", test=False):
        """Add a plot for the corresponding index

        :param plot_num: The linear index of the plot
        :type plot_num: int

        :param data_frame: The data for the plot
        :type data_frame: :mod:`pandas.DataFrame`

        :param title: The title for the plot
        :type title: str
        """

        datapoints = sum(len(v) for _, v in data_frame.iteritems())
        if datapoints > self._attr["max_datapoints"]:
            msg = "This plot is too big and will probably make your browser unresponsive.  If you are happy to wait, pass max_datapoints={} to view()".\
                  format(datapoints + 1)
            raise ValueError(msg)

        fig_name = self._fig_map[plot_num]
        fig_params = {}
        fig_params["data"] = df_to_dygraph(data_frame)
        fig_params["name"] = fig_name
        fig_params["rangesel"] = False
        fig_params["logscale"] = False
        fig_params["title"] = title
        fig_params["step_plot"] = self._attr["step_plot"]
        fig_params["fill_graph"] = self._attr["fill"]
        if "fill_alpha" in self._attr:
            fig_params["fill_alpha"] = self._attr["fill_alpha"]
            fig_params["fill_graph"] = True
        fig_params["per_line"] = self._attr["per_line"]
        fig_params["height"] = self._attr["height"]

        self._check_add_scatter(fig_params)

        if "group" in self._attr:
            fig_params["syncGroup"] = self._attr["group"]
            if "sync_zoom" in self._attr:
                fig_params["syncZoom"] = self._attr["sync_zoom"]
            else:
                fig_params["syncZoom"] = AttrConf.DEFAULT_SYNC_ZOOM

        if "ylim" in self._attr:
            fig_params["valueRange"] = self._attr["ylim"]

        if "xlim" in self._attr:
            fig_params["dateWindow"] = self._attr["xlim"]

        fig_data = "var {}_data = {};".format(fig_name, json.dumps(fig_params))

        self._js_plot_data.append("<script>")
        self._js_plot_data.append(fig_data)
        self._js_plot_data.append("</script>")

    def finish(self):
        """Called when the Plotting is finished"""

        display(HTML(self.html()))

    def html(self):
        """Return the raw HTML text"""

        return "\n".join(self._html + self._js_plot_data + self._js)
