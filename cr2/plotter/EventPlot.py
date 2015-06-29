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
# File:        EventPlot.py
# ----------------------------------------------------------------
# $
#
"""
The EventPlot is used to represent Events with two characteristics:

    * A name, which determines the colour on the plot
    * A lane, which determines the lane in which the event occurred

In the case of a cpu residency plot, the term lane can be equated to
a CPU and the name attribute can be the PID of the task
"""

from cr2.plotter import AttrConf
import uuid
import json
import os
from IPython.display import display, HTML
from cr2.plotter.AbstractDataPlotter import AbstractDataPlotter

if not AttrConf.PLOTTER_IPYTHON:
    raise ImportError("Ipython Environment not Found")

# pylint: disable=R0201
# pylint: disable=R0921
# Initialize Resources
from cr2.plotter import Utils
Utils.iplot_install("EventPlot")


class EventPlot(AbstractDataPlotter):

    """EventPlot Class that extends
       AbstractDataPlotter"""

    def __init__(self, data, keys, lane_prefix, num_lanes, summary=True):
        """
            Args:
                data: Data of the format:
                    [ {"id"   : <id>,
                       "name" : <name>,
                       "lane" : <lane_number>
                       "start": <event_start_time
                       "end"  : <event_end_time>
                      },
                      .
                      .
                      .
                    ]
                keys: List of unique names in the data dictionary
                lane_prefix: A string prefix to be used to name each lane
                num_lanes: Total number of expected lanes
        """

        self._fig_name = self._generate_fig_name
        self._html = []
        self._fig_name = self._generate_fig_name()

        graph = {}
        graph["data"] = data
        graph["keys"] = keys
        graph["lanes"] = self._get_lanes(lane_prefix, num_lanes)
        graph["showSummary"] = summary

        # Write the graph data to the JSON File
        json_file = os.path.join(
            AttrConf.PLOTTER_STATIC_DATA_DIR,
            self._fig_name +
            ".json")
        with open(json_file, "w") as json_fh:
            json.dump(graph, json_fh)

        # Initialize the HTML, CSS and JS Components
        self._add_css()
        self._init_html()

    def view(self):
        """Views the Graph Object"""
        display(HTML(self.html()))

    def savefig(self, path):
        """Save the plot in the provided path"""

        raise NotImplementedError(
            "Save is not currently implemented for EventPlot")

    def _get_lanes(self, lane_prefix, num_lanes):
        """Populate the lanes for the plot"""

        lanes = []
        for idx in range(num_lanes):
            lanes.append({"id": idx, "label": "{}{}".format(lane_prefix, idx)})
        return lanes

    def _generate_fig_name(self):
        """Generate a unqiue name for the figure"""

        fig_name = "fig_" + uuid.uuid4().hex
        return fig_name

    def _init_html(self):
        """Initialize HTML for the plot"""
        div_js = """
        <script>
            var req = require.config( {

                baseUrl: "/static/plotter_scripts",
                shim: {
                    "EventPlot/d3.tip.v0.6.3": ["EventPlot/d3.v3.min"],
                    "EventPlot/EventPlot": {

                        "deps": ["EventPlot/d3.v3.min", "EventPlot/d3.tip.v0.6.3" ],
                        "exports":  "EventPlot"
                    }
                }
            });
            req(["require", "EventPlot/EventPlot"], function() {
               EventPlot.generate('""" + self._fig_name + """');
            });
        </script>
        """

        self._html.append(
            '<div id="{}" class="eventplot">{}</div>'.format(self._fig_name,
                                                             div_js))

    def _add_css(self):
        """Append the CSS to the HTML code generated"""

        base_dir = os.path.dirname(os.path.realpath(__file__))
        css_file = os.path.join(base_dir, "css/EventPlot.css")
        css_fh = open(css_file, 'r')
        self._html.append("<style>")
        self._html += css_fh.readlines()
        self._html.append("</style>")
        css_fh.close()

    def html(self):
        """Return a Raw HTML string for the plot"""

        return "\n".join(self._html)
