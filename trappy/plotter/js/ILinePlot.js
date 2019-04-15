/*
 *    Copyright 2015-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var ILinePlot = ( function() {

   var graphs = new Array();
   var syncObjs = new Array();

    var purge = function() {
        for (var div_name in graphs) {
            if (document.getElementById(div_name) == null) {
                delete graphs[div_name];
            }
        }
    };

    var sync = function(group) {

        var syncGraphs = Array();
        var xRange;
        var yRange;
        var syncZoom = true;

        for (var div_name in graphs) {

            if (graphs[div_name].group == group) {
                syncGraphs.push(graphs[div_name].graph);
                syncZoom = syncZoom & graphs[div_name].syncZoom;

                var xR = graphs[div_name].graph.xAxisRange();
                var yR = graphs[div_name].graph.yAxisRange();

                if (xRange != undefined) {
                    if (xR[0] < xRange[0])
                        xRange[0] = xR[0];
                    if (xR[1] > xRange[1])
                        xRange[1] = xR[1];
                } else
                    xRange = xR;

                if (yRange != undefined) {
                    if (yR[0] < yRange[0])
                        yRange[0] = yR[0];
                    if (yR[1] > yRange[1])
                        yRange[1] = yR[1];
                } else
                    yRange = yR;
            }
        }

        if (syncGraphs.length >= 2) {
            if (syncZoom) {
                if (syncObjs[group] != undefined)
                    syncObjs[group].detach();

                syncObjs[group] = Dygraph.synchronize(syncGraphs, {
                    zoom: true,
                    selection: false,
                    range: true
                });
            }

            $.each(syncGraphs, function(g) {
                var graph = syncGraphs[g];

                graph.updateOptions({
                    valueRange: yRange,
                    dateWindow: xRange
                });

                if (graph.padFront_ == undefined) {
                    graph.padFront_ = true;
                    var _decoy_elem = new Array(graph.rawData_[0].length);
                    graph.rawData_.unshift(_decoy_elem);
                }
                graph.rawData_[0][0] = xRange[0];

                if (graph.padBack_ == undefined) {
                    graph.padBack_ = true;
                    var _decoy_elem = new Array(graph.rawData_[0].length);
                    graph.rawData_.push(_decoy_elem);
                }
                graph.rawData_[graph.rawData_.length - 1][0] = xRange[1];
            });
        }
    };

    var generate = function(data, colors) {
        create_graph(data, colors);
        purge();
        if (data.syncGroup != undefined)
            sync(data.syncGroup);
    };

    var create_graph = function(t_info, colors) {
        var tabular = t_info.data;

        var options = {
            legend: 'always',
            title: t_info.title,
            labels: tabular.labels,
            labelsDivStyles: {
                'textAlign': 'right'
            },
            rollPeriod: 1,
            animatedZooms: true,
            connectSeparatedPoints: true,
            showRangeSelector: t_info.rangesel,
            rangeSelectorHeight: 50,
            stepPlot: t_info.step_plot,
            logscale: t_info.logscale,
            fillGraph: t_info.fill_graph,
            labelsDiv: t_info.name + "_legend",
            errorBars: false,
            labelsSeparateLines: true,
            valueRange: t_info.valueRange,
            drawPoints: t_info.drawPoints,
            strokeWidth: t_info.strokeWidth,
            pointSize: t_info.pointSize,
            dateWindow: t_info.dateWindow
        };

        if (typeof t_info.fill_alpha !== 'undefined')
            options.fillAlpha = t_info.fill_alpha;

        if (typeof colors !== 'undefined')
            options["colors"] = colors;

        var graph = new Dygraph(document.getElementById(t_info.name), tabular.data, options);

        var width = $("#" + t_info.name)
            .closest(".output_subarea").width() / t_info.per_line

        /*
         * Remove 3 pixels from width to avoid unnecessary horizontal scrollbar
         */
        graph.resize(width - 3, t_info.height);

        $(window).on("resize." + t_info.name, function() {

            var width = $("#" + t_info.name)
                .closest(".output_subarea").width() / t_info.per_line

            graph.resize(width, t_info.height);
        });

        graphs[t_info.name] =
            {
                graph: graph,
                group: t_info.syncGroup,
                syncZoom: t_info.syncZoom
            };

    };

    return {
        generate: generate
    };

}());
