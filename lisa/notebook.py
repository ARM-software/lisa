# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2019, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Various utilities for interactive notebooks.
"""

from ipywidgets import widgets, Output, HBox, Layout
from IPython.display import display

def axis_link_dataframes(axis, df_list, before=1, after=5, cursor_color='red', follow_cursor=False):
    """
    Link some dataframes to an axis displayed in the interactive matplotlib widget.


    :param axis: Axis to link to.
    :type axis: matplotlib.axes.Axes

    :param df_list: List of pandas dataframe to link.
    :type df_list: list(pandas.DataFrame)

    :param before: Number of dataframe rows to display before the selected
        location.
    :type before: int

    :param after: Number of dataframe rows to display after the selected
        location.
    :type after: int

    :param cursor_color: Color of the vertical line added at the clicked
        location.
    :type cursor_color: str

    :param follow_cursor: If ``True``, the cursor will be followed without the
        need to click.
    :type follow_cursor: bool

    When the user clicks on the graph, a vertical marker will appear and the
    dataframe slice will update to show the relevant row.

    .. note:: This requires the matplotlib widget enabled using ``%matplotlib
        widget`` magic.
    """
    output_list = [widgets.Output() for df in df_list]
    layout = Layout(
        # Overflow items to the next line rather than hiding them
        flex_flow='row wrap',
        # Evenly spread on one line of item when there is more than one item,
        # align left otherwise
        justify_content='space-around' if len(df_list) > 1 else 'flex-start',
    )
    hbox = widgets.HBox(output_list, layout=layout)
    cursor_vline = axis.axvline(color=cursor_color)

    def show_loc(loc):
        cursor_vline.set_xdata(loc)

        for df, output in zip(df_list, output_list):
            if loc < df.index[0]:
                iloc = 0
            elif loc > df.index[-1]:
                iloc = -1
            else:
                iloc = df.index.get_loc(loc, method='ffill')
            index_loc = df.index[iloc]

            begin = max(iloc - before, 0)
            end = min(iloc + after, len(df))
            sliced_df = df.iloc[begin:end]

            def highlight_row(row):
                if row.name == index_loc:
                    return ['background: lightblue'] * len(row)
                else:
                    return [''] * len(row)

            styler = sliced_df.style.apply(highlight_row, axis=1)
            styler = styler.set_properties(**{
                'text-align': 'left',
                # perserve multiple consecutive spaces
                'white-space': 'pre',
                # Make sure all chars have the same width to preserve column
                # alignments in preformatted strings
                'font-family': 'monospace',
            })


            # wait=True avoids flicker by waiting for new content to be ready
            # to display before clearing the previous one
            output.clear_output(wait=True)
            with output:
                display(styler)

    init_loc = min(df.index[0] for df in df_list)
    show_loc(init_loc)

    def handler(event):
        loc = event.xdata
        return show_loc(loc)

    event = 'motion_notify_event' if follow_cursor else 'button_press_event'
    axis.get_figure().canvas.mpl_connect('button_press_event', handler)
    display(hbox)


# vim :set tabstop=4 shiftwidth=4 textwidth=80 expandtab
