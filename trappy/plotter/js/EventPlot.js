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

var EventPlot = (function () {

    /* EventPlot receives data that is hashed by the keys
     * and each element in the data is sorted by start time.
     * Since events on each lane are mutually exclusive, they
     * they are also sorted by the end time. We use this information
     * and binary search on the input data for filtering events
     * This maintains filtering complexity to O[KLogN]
     */

    var GUIDER_WIDTH = 2;

    infoProps = {
        START_GUIDER_COLOR: "green",
        END_GUIDER_COLOR: "red",
        DELTA_COLOR: "blue",
        GUIDER_WIDTH: 2,
        TOP_MARGIN: 20,
        HEIGHT: 30,
        START_PREFIX: "A = ",
        END_PREFIX: "B = ",
        DELTA_PREFIX: "A - B = ",
        XPAD: 10,
        YPAD: 5,
        BOX_BUFFER: 2,
        BOX_WIDTH_RATIO: 0.6
    }

    var search_data = function (data, key, value, left, right) {

        var mid;

        while (left < right) {

            mid = Math.floor((left + right) / 2)
            if (data[mid][key] > value)
                right = mid;
            else
                left = mid + 1;
        }
        return left;
    }


    /* Return the information for the current process
     * pointed by the mouse
     */
    var getCurrentInfo = function(ePlot, x0, y0) {

        for (name in ePlot.items) {

            var data = ePlot.items[name];
            var xMax = ePlot.zoomScale.domain()[1];
            var right = search_data(data, 0, xMax, 0, data.length - 1);
            var left = search_data(data, 1, x0, 0, right);

            if (data) {
                var candidate = data[left];
                if (candidate[0] <= x0 &&
                        candidate[1] >= x0 &&
                        candidate[2] == y0)
                    return {
                            name: name,
                            info: candidate
                        };
            }
        }
    }

    var generate = function (div_name, base, chart_data) {

        var margin, brush, x, ext, yMain, chart, main,
            mainAxis,
            itemRects, items, colourAxis, tip, lanes;

        var process_chart_data = function (d) {
            items = d.data;
            lanes = d.lanes;
            var names = d.keys;
            var showSummary = d.showSummary;
            var div = $("#" + div_name);

            margin = {
                    top: 15,
                    right: 15,
                    bottom: 15,
                    left: 70
                }, width = div.width() - margin.left - margin.right,

                mainHeight = 50 * lanes.length - margin.top - margin.bottom;

            x = d3.scale.linear()
                .domain(d.xDomain)
                .range([0, width]);

            var zoomScale = d3.scale.linear()
                .domain(d.xDomain)
                .range([0, width]);

            var xMin = x.domain()[0];
            var xMax = x.domain()[1];

            if (!d.colorMap) {
                // Colour Ordinal scale. Uses Category20 Colors
                colours = d3.scale.category20().range();
            } else {
                // Use colours provided by user
                var colours = [];
                for (var i in names)
                    if (names[i] in d.colorMap)
                        colours.push(d.colorMap[names[i]]);
            }
            colourAxis = d3.scale.ordinal()
                .range(colours)
                .domain(names);

            brushScale = d3.scale.linear()
                .range([0, width]);
            ext = d3.extent(lanes, function (d) {
                return d.id;
            });
            yMain = d3.scale.linear()
                .domain([ext[0], ext[1] +
                    1
                ])
                .range([0, mainHeight]);


            var ePlot;


            $("#" + div_name)
                .append('<div class="pull-right">' +
                    '<button type="button" class="btn btn-sm btn-info" ' +
                    'onclick="EventPlot.create_help_dialog(' + base +
                    ')">Help</button></div>')

            var iDesc = drawInfo(div_name, margin, width);

            chart = d3.select('#' + div_name)
                .append('svg:svg')
                .attr('width', width + margin.right +
                    margin.left)
                .attr('height', mainHeight + margin.top +
                    margin.bottom + 5)
                .attr('class', 'chart')


            main = chart.append('g')
                .attr('transform', 'translate(' + margin.left +
                    ',' + margin.top + ')')
                .attr('width', width)
                .attr('height', mainHeight)
                .attr('class', 'main')

            main.append('g')
                .selectAll('.laneLines')
                .data(lanes)
                .enter()
                .append('line')
                .attr('x1', 0)
                .attr('y1', function (d) {
                    return d3.round(yMain(d.id)) + 0.5;
                })
                .attr('x2', width)
                .attr('y2', function (d) {
                    return d3.round(yMain(d.id)) + 0.5;
                })
                .attr('stroke', function (d) {
                    return d.label === '' ? 'white' :
                        'lightgray'
                });

            main.append('g')
                .selectAll('.laneText')
                .data(lanes)
                .enter()
                .append('text')
                .attr('x', 0)
                .text(function (d) {
                    return d.label;
                })
                .attr('y', function (d) {
                    return yMain(d.id + .5);
                })
                .attr('dy', '0.5ex')
                .attr('text-anchor', 'end')
                .attr('class', 'laneText');

            mainAxis = d3.svg.axis()
                .scale(brushScale)
                .orient('bottom');

            tip = d3.tip()
                .attr('class', 'd3-tip')
                .html(function (d) {
                    return "<span style='color:white'>" +
                        d.name + "</span>";
                })

            main.append('g')
                .attr('transform', 'translate(0,' +
                    mainHeight + ')')
                .attr('class', 'main axis')
                .call(mainAxis);

            var ePlot;

            ePlot = {
                div: div,
                div_name: div_name,
                margin: margin,
                chart: chart,
                mainHeight: mainHeight,
                width: width,
                x: x,
                brushScale: brushScale,
                ext: ext,
                yMain: yMain,
                main: main,
                mainAxis: mainAxis,
                items: items,
                colourAxis: colourAxis,
                tip: tip,
                lanes: lanes,
                names: names,
                iDesc: iDesc,
            };
            ePlot.zoomScale = zoomScale;

            if (showSummary)
                drawMini(ePlot);

            var outgoing;
            var zoomed = function () {

                if (zoomScale.domain()[0] < xMin) {
                    zoom.translate([zoom.translate()[
                            0] - zoomScale(
                            xMin) +
                        zoomScale.range()[0],
                        zoom.translate()[
                            1]
                    ]);
                } else if (zoomScale.domain()[1] >
                    xMax) {
                    zoom.translate([zoom.translate()[
                            0] - zoomScale(
                            xMax) +
                        zoomScale.range()[1],
                        zoom.translate()[
                            1]
                    ]);

                }

                outgoing = main.selectAll(".mItem")
                    .attr("visibility", "hidden");
                drawMain(ePlot, zoomScale.domain()[0],
                    zoomScale.domain()[1]);
                if (showSummary) {
                    brush.extent(zoomScale.domain());
                    ePlot.mini.select(".brush")
                        .call(
                            brush);
                }

                brushScale.domain(zoomScale.domain());
                ePlot.main.select('.main.axis')
                    .call(ePlot.mainAxis)

                updateInfo(ePlot);
            };

            var rightClickCtrlAltHandler = function(x0, y0) {

                x0 = ePlot.zoomScale.invert(x0);
                y0 = Math.floor(ePlot.yMain.invert(y0));
                var current = getCurrentInfo(ePlot, x0, y0);

                if (current) {
                    ePlot.iDesc.currentProc.text(current.name)
                    ePlot.iDesc.currentInfo.text(
                        current.info[0].toFixed(6)
                        + " to " +
                        current.info[1].toFixed(6) +
                        " (" + (current.info[1] - current.info[0])
                        .toFixed(6) + ")")

                    removeContextRect(ePlot);
                    ePlot.contextRect = drawContextRect(ePlot, current.info[0], current.info[1], current.info[2], true)
                    ePlot.iDesc.currentDisp.attr("stroke", ePlot.colourAxis(current.name));
                }
            }

            var contextMenuHandler = function() {

                var e = d3.event;
                var x0 = d3.mouse(this)[0] - ePlot.margin.left;
                var y0 = d3.mouse(this)[1] - ePlot.margin.top;

                if (e.ctrlKey && e.altKey)
                    rightClickCtrlAltHandler(x0, y0);

                else if (e.ctrlKey) {

                    if (ePlot.endGuider)
                        ePlot.endGuider = ePlot.endGuider.remove();

                    ePlot.endGuider = drawVerticalLine(ePlot, x0,
                        infoProps.END_GUIDER_COLOR, "B");
                    ePlot.endGuider._x_pos = ePlot.zoomScale.invert(x0);
                    iDesc.endText.text(infoProps.END_PREFIX + ePlot.endGuider._x_pos.toFixed(6))

                } else {

                    if (ePlot.startGuider)
                        ePlot.startGuider = ePlot.startGuider.remove();

                    ePlot.startGuider = drawVerticalLine(ePlot, x0,
                        infoProps.START_GUIDER_COLOR, "A");
                    ePlot.startGuider._x_pos = ePlot.zoomScale.invert(x0);
                    iDesc.startText.text(infoProps.START_PREFIX + ePlot.startGuider._x_pos.toFixed(6))
                }

                if (ePlot.endGuider && ePlot.startGuider)
                    iDesc.deltaText.text(infoProps.DELTA_PREFIX +
                            (ePlot.endGuider._x_pos - ePlot.startGuider._x_pos)
                            .toFixed(6)
                        )

                d3.event.preventDefault();
            }

            chart.on("contextmenu", contextMenuHandler);

            if (showSummary) {
                var _brushed_event = function () {
                    main.selectAll("path")
                        .remove();
                    var brush_xmin = brush.extent()[0];
                    var brush_xmax = brush.extent()[1];

                    var t = zoom.translate(),
                        new_domain = brush.extent(),
                        scale;

                    /*
                     *    scale = x.range()[1] - x.range[0]
                     *          --------------------------
                     *          x(x.domain()[1] - x.domain()[0])
                     *
                     *                             _                                   _
                     *  new_domain[0] =  x.invert | x.range()[0]  -   z.translate()[0]  |
                     *                            |                 ------------------- |
                     *                            |_                     z.scale()     _|
                     *
                     *
                     *
                     *  translate[0] = x.range()[0] - x(new_domain[0])) * zoom.scale()
                     */

                    scale = (width) / x(x.domain()[0] +
                        new_domain[1] -
                        new_domain[0]);
                    zoom.scale(scale);
                    t[0] = x.range()[0] - (x(new_domain[
                        0]) * scale);
                    zoom.translate(t);


                    brushScale.domain(brush.extent())
                    drawMain(ePlot, brush_xmin,
                        brush_xmax);
                    ePlot.main.select('.main.axis')
                        .call(ePlot.mainAxis)

                    updateInfo(ePlot);
                };

                brush = d3.svg.brush()
                    .x(x)
                    .extent(x.domain())
                    .on("brush", _brushed_event);

                ePlot.mini.append('g')
                    .attr('class', 'brush')
                    .call(brush)
                    .selectAll('rect')
                    .attr('y', 1)
                    .attr('height', ePlot.miniHeight - 1);
            }

            var zoom = d3.behavior.zoom()
                .x(zoomScale)
                .on(
                    "zoom", zoomed)
                .on("zoomend", function () {
                    if (outgoing)
                        outgoing.remove()
                })
                .scaleExtent([1, 4096]);
            chart.call(zoom);

            drawMain(ePlot, xMin, xMax);
            ePlot.main.select('.main.axis')
                .call(ePlot.mainAxis)

            var resize = function() {

                var width = div.width() - margin.left
                    - margin.right;

                /* Update scale ranges */
                x.range([0, width]);
                zoomScale.range([0, width]);
                brushScale.range([0, width]);
                ePlot.width = width;

                resize_main(ePlot);
                resize_info(ePlot);
                resize_mini(ePlot);
                zoomed();

            }

            d3.select(window)
                .on("resize." + ePlot.div_name, resize)

            return ePlot;

        }

        /*
         * If chart_data is passed, process data directly
         */
        process_chart_data(chart_data);
    };


    var resize_mini = function(ePlot) {

        d3.select(ePlot.mini.node().parentNode)
            .attr("width", ePlot.div.width());
        ePlot.iDesc.info_svg
            .attr("width", ePlot.div.width());
        ePlot.mini.selectAll("line")
            .attr("x2", ePlot.width);
        ePlot.mini.call(ePlot.miniAxis);
        ePlot.mini.selectAll(".miniItem").remove();
        drawMiniPaths(ePlot);
    }

    var resize_main = function(ePlot) {

        d3.select(ePlot.main.node().parentNode)
            .attr("width", ePlot.div.width());
        ePlot.main.selectAll("line")
            .attr("x2", ePlot.width);
    }

    var resize_info = function(ePlot) {

        var width_box_one = infoProps.BOX_WIDTH_RATIO * ePlot.width;
        var width_box_two = ePlot.width - width_box_one;

        ePlot.iDesc.info
            .attr("width", width);
        ePlot.iDesc.guiderInfo
            .attr("width", width_box_one - infoProps.BOX_BUFFER);
        ePlot.iDesc.currentDisp
            .attr("x", width_box_one + infoProps.BOX_BUFFER);
        ePlot.iDesc.currentDisp
            .attr("width", width_box_two - infoProps.BOX_BUFFER);
        ePlot.iDesc.deltaText
            .attr("x", (width_box_one / 2) - infoProps.XPAD)
        ePlot.iDesc.endText
            .attr("x", width_box_one - infoProps.XPAD)
        ePlot.iDesc.currentProc
            .attr("x", width_box_one + infoProps.XPAD + infoProps.BOX_BUFFER)
        ePlot.iDesc.currentInfo
            .attr("x", ePlot.width - infoProps.XPAD)
    }

    var drawInfo = function (div_name, margin, width) {

        var infoHeight = 30;
        var _top = 20;
        var LINE_WIDTH = 2

        var iDesc = {};

        var width_box_one = infoProps.BOX_WIDTH_RATIO * width;
        var width_box_two = width - width_box_one

        iDesc.info_svg = d3.select("#" + div_name)
            .append(
                "svg:svg")
            .attr('width', width + margin.right +
                margin.left)
            .attr('height', infoHeight + infoProps.TOP_MARGIN + LINE_WIDTH)
            .attr('class', 'info')

        iDesc.info = iDesc.info_svg.append("g")
            .attr("transform", "translate(" + margin.left +
                 "," + infoProps.TOP_MARGIN + ")")
            .attr('width', width)
            .attr("class", "main")
            .attr('height', infoProps.HEIGHT)

        iDesc.guiderInfo = iDesc.info.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", width_box_one - infoProps.BOX_BUFFER)
            .attr("height", infoHeight)
            .attr("stroke", "lightgray")
            .attr("fill", "none")
            .attr("stroke-width", 1);

        iDesc.currentDisp = iDesc.info.append("rect")
            .attr("x", width_box_one + infoProps.BOX_BUFFER)
            .attr("y", 0)
            .attr("width", width_box_two - infoProps.BOX_BUFFER)
            .attr("height", infoHeight)
            .attr("stroke", "lightgray")
            .attr("fill", "none")
            .attr("stroke-width", 1);

       iDesc.startText = iDesc.info.append("text")
            .text("")
            .attr("x", infoProps.XPAD)
            .attr("y", infoProps.HEIGHT / 2 + infoProps.YPAD)
            .attr("fill", infoProps.START_GUIDER_COLOR);

       iDesc.deltaText = iDesc.info.append("text")
            .text("")
            .attr("x", (width_box_one / 2) - infoProps.XPAD)
            .attr("y", infoProps.HEIGHT / 2 + infoProps.YPAD)
            .attr("fill", infoProps.DELTA_COLOR);

       iDesc.endText = iDesc.info.append("text")
            .text("")
            .attr("x", width_box_one - infoProps.XPAD)
            .attr("text-anchor", "end")
            .attr("y", infoProps.HEIGHT / 2 + infoProps.YPAD)
            .attr("fill", infoProps.END_GUIDER_COLOR);

        iDesc.currentProc = iDesc.info.append("text")
            .text("")
            .attr("x", width_box_one + infoProps.XPAD + infoProps.BOX_BUFFER)
            .attr("text-anchor", "start")
            .attr("y", infoProps.HEIGHT / 2 + infoProps.YPAD)

        iDesc.currentInfo = iDesc.info.append("text")
            .text("")
            .attr("x", width - infoProps.XPAD)
            .attr("text-anchor", "end")
            .attr("y", infoProps.HEIGHT / 2 + infoProps.YPAD)

        return iDesc;

    }

    var drawVerticalLine = function (ePlot, x, color, text) {

        var line = ePlot.main.append("g")

        line.append("line")
            .style("stroke", color)
            .style("stroke-width", GUIDER_WIDTH)
            .attr("x1", x)
            .attr("x2", x)
            .attr("y1", 0)
            .attr("y2", ePlot.mainHeight)

        line.append("text")
            .text(text)
            .attr("y", -1)
            .attr("x", x)
            .attr("text-anchor", "middle")
            .attr("fill", color)

        return line;
    };

    var removeContextRect = function(ePlot) {
        if (ePlot.contextRect && ePlot.contextRect.rect)
            ePlot.contextRect.rect.remove();
    }

    var drawContextRect = function (ePlot, x0, x1, y, animate) {

        var xMin = ePlot.zoomScale.domain()[0];
        var xMax = ePlot.zoomScale.domain()[1];
        var bounds = [Math.max(x0, xMin), Math.min(x1,
                xMax)]

        if (bounds[0] >= bounds[1])
            return {
                rect: false,
                x0: x0,
                x1: x1,
                y: y,
            }

        var rect = ePlot.main.selectAll(".contextRect").data([""])

        if (animate)
            rect.enter().append("rect")
                .attr("x", ePlot.zoomScale(bounds[0]))
                .attr("y", ePlot.yMain(y))
                .attr("height", ePlot.yMain(1))
                .attr("class", "contextRect")
                .attr("width", 0)
                .transition()
                .attr("width", ePlot.zoomScale(bounds[1]) - ePlot.zoomScale(bounds[0]))
        else
            rect.enter().append("rect")
                .attr("x", ePlot.zoomScale(bounds[0]))
                .attr("y", ePlot.yMain(y))
                .attr("class", "contextRect")
                .attr("height", ePlot.yMain(1))
                .attr("width", ePlot.zoomScale(bounds[1]) - ePlot.zoomScale(bounds[0]))

        return {
                rect: rect,
                x0: x0,
                x1: x1,
                y: y,
        }
    }

    var checkGuiderRange = function (ePlot, xpos) {

        if (xpos >= ePlot.zoomScale.domain()[0] &&
            xpos <= ePlot.zoomScale.domain()[1])
            return true;
        else
            return false;
    }

    var updateInfo = function (ePlot) {

        if (ePlot.endGuider) {

            var xpos = ePlot.endGuider._x_pos;
            ePlot.endGuider.remove();

            if (checkGuiderRange(ePlot, xpos)) {
                ePlot.endGuider = drawVerticalLine(ePlot, ePlot.zoomScale(xpos),
                    infoProps.END_GUIDER_COLOR, "B");
                ePlot.endGuider._x_pos = xpos;
            }
        }

        if (ePlot.startGuider) {

            var xpos = ePlot.startGuider._x_pos;
            ePlot.startGuider.remove();

            if (checkGuiderRange(ePlot, xpos)) {
                ePlot.startGuider = drawVerticalLine(ePlot, ePlot.zoomScale(xpos),
                    infoProps.START_GUIDER_COLOR, "A");
                ePlot.startGuider._x_pos = xpos
            }
        }

        if (ePlot.contextRect) {
            removeContextRect(ePlot);
            ePlot.contextRect = drawContextRect(ePlot, ePlot.contextRect.x0,
                                    ePlot.contextRect.x1,
                                    ePlot.contextRect.y,
                                    false);
        }

    }

    var drawMiniPaths = function(ePlot) {

        ePlot.mini.append('g')
            .selectAll('miniItems')
            .data(getPaths(ePlot, ePlot.x, ePlot.yMini))
            .enter()
            .append('path')
            .attr('class', function (d) {
                return 'miniItem'
            })
            .attr('d', function (d) {
                return d.path;
            })
            .attr("stroke", function (d) {
                return d.color
            })
            .attr("class", "miniItem");
    }

    var drawMini = function (ePlot) {

        var miniHeight = ePlot.lanes.length * 12 + 50;

        var miniAxis = d3.svg.axis()
            .scale(ePlot.x)
            .orient('bottom');

        var yMini = d3.scale.linear()
            .domain([ePlot.ext[0], ePlot.ext[1] +
                1
            ])
            .range([0, miniHeight]);

        ePlot.yMini = yMini;
        ePlot.miniAxis = miniAxis;
        ePlot.miniHeight = miniHeight;

        var summary = d3.select("#" + ePlot.div_name)
            .append(
                "svg:svg")
            .attr('width', ePlot.width + ePlot.margin.right +
                ePlot.margin.left)
            .attr('height', miniHeight + ePlot.margin.bottom +
                ePlot.margin.top)
            .attr('class', 'chart')

        var mini = summary.append('g')
            .attr("transform", "translate(" + ePlot.margin.left +
                "," + ePlot.margin.top + ")")
            .attr('width', ePlot.width)
            .attr('height', ePlot.miniHeight)
            .attr('class', 'mini');

        mini.append('g')
            .selectAll('.laneLines')
            .data(ePlot.lanes)
            .enter()
            .append('line')
            .attr('x1', 0)
            .attr('y1', function (d) {
                return d3.round(ePlot.yMini(d.id)) + 0.5;
            })
            .attr('x2', ePlot.width)
            .attr('y2', function (d) {
                return d3.round(ePlot.yMini(d.id)) + 0.5;
            })
            .attr('stroke', function (d) {
                return d.label === '' ? 'white' :
                    'lightgray'
            });

        mini.append('g')
            .attr('transform', 'translate(0,' +
                ePlot.miniHeight + ')')
            .attr('class', 'axis')
            .call(ePlot.miniAxis);

        ePlot.mini = mini
        drawMiniPaths(ePlot);

        mini.append('g')
            .selectAll('.laneText')
            .data(ePlot.lanes)
            .enter()
            .append('text')
            .text(function (d) {
                return d.label;
            })
            .attr('x', -10)
            .attr('y', function (d) {
                return ePlot.yMini(d.id + .5);
            })
            .attr('dy', '0.5ex')
            .attr('text-anchor', 'end')
            .attr('class', 'laneText');

        return mini;
    };


    var drawMain = function (ePlot, xMin, xMax) {

        var rects, labels;
        var dMin = 10000;
        var paths = getPaths(ePlot, ePlot.zoomScale, ePlot.yMain);
        ePlot.brushScale.domain([xMin, xMax]);

        if (paths.length == 0)
            return;

        ePlot.main
            .selectAll('mainItems')
            .data(paths)
            .enter()
            .append('path')
            .attr("shape-rendering", "crispEdges")
            .attr('d', function (d) {
                return d.path;
            })
            .attr("class", "mItem")
            .attr("stroke-width", function(d) {
               return  0.8 * ePlot.yMain(1);
            })
            .attr("stroke", function (d) {
                return d.color
            })
            .call(ePlot.tip)
            .on("mouseover", ePlot.tip.show)
            .on('mouseout', ePlot.tip.hide)
            .on('mousemove', function () {
                var xDisp = parseFloat(ePlot.tip.style("width")) /
                    2.0
                ePlot.tip.style("left", (d3.event.pageX - xDisp) +
                        "px")
                    .style("top", Math.max(0, d3.event.pageY -
                        47) + "px");
            })
    };


   function  _handle_equality(d, xMin, xMax, x, y, lane) {
        var offset = 0.5 * y(1) + 0.5
        var bounds = [Math.max(d[0], xMin), Math.min(d[1],
            xMax)]
        if (bounds[0] < bounds[1])
            return 'M' + ' ' + x(bounds[0]) + ' ' + (y(lane) + offset) + ' H ' +  x(bounds[1]);
        else
            return '';
    };

    function _process(path, d, xMin, xMax, x, y, offset, lane) {

        var start = d[0];
        if (start < xMin)
            start = xMin;
        var end = d[1];
        if (end > xMax)
            end = xMax;

        start = x(start);
        end = x(end);

        if ((end - start) < 0.01)
            return path;
        else if ((end - start) < 1)
            end = start + 1;

        path += 'M' + ' ' + start + ' ' + (y(lane) + offset) + ' H ' +  end;
        return path;
    }

    var _get_path = function(new_data, xMin, xMax, offset, x, y, stride) {

            var path = ''
            var max_rects = 2000;

            for (var lane in new_data) {
                var data = new_data[lane];
                var right = search_data(data, 0, xMax, 0, data.length - 1)
                var left = search_data(data, 1, xMin, 0, right)

                //Handle Equality
                if (left == right)
                    path += _handle_equality(data[left], xMin, xMax, x, y, lane);

                data = data.slice(left, right + 1);

                var stride_length = 1;
                if (stride)
                    stride_length = Math.max(Math.ceil(data.length / max_rects), 1);

                for (var i = 0; i < data.length; i+= stride_length)
                    path = _process(path, data[i], xMin, xMax, x, y, offset, lane);
        }

        return path;
    }

    var getPaths = function (ePlot, x, y, stride) {

        var keys = ePlot.names;
        var items = ePlot.items;
        var colourAxis = ePlot.colourAxis;

        var xMin = x.domain()[0];
        var xMax = x.domain()[1];
        var paths = {},
            d, offset = 0.5 * y(1) + 0.5,
            result = [];

        for (var i in keys) {
            var name = keys[i];
            var path = _get_path(items[name], xMin, xMax, offset, x, y, stride)
            /* This is critical. Adding paths for non
             * existent processes in the window* can be
             * very expensive as there is one SVG per process
             * and SVG rendering is expensive
             */
            if (!path || path == "")
                continue

            result.push({
                color: colourAxis(name),
                path: path,
                name: name
            });
        }

        return result;

    }

    var create_dialog_body = function (body, title) {

        var element = $("<div/>")
            .addClass("modal fade")
            .attr("role", "dialog")
            .attr("tabindex", -1)

        element.append(
            $("<div/>")
            .addClass("modal-dialog")
            .attr("role", "document")
            .append(
                $("<div/>")
                .addClass("modal-content")
                .append(
                    $("<div/>")
                    .addClass("modal-header")
                    .append(
                        $("<button>")
                        .addClass("close")
                        .attr("data-dismiss",
                            "modal")
                        .append($("<span/>")
                            .html("&times;"))
                    )
                    .append($("<h4/>")
                        .addClass("modal-title")
                        .text(title)
                    )
                    .append($("<div/>")
                        .addClass("modal-body")
                        .append(body)
                    )
                    .append(
                        $("<div/>")
                        .addClass("modal-footer")
                        .append(
                            $("<button>")
                            .addClass("btn btn-default")
                            .attr("data-dismiss", "modal")
                            .text("Close")
                        )
                    )
                )
            )
        )

        return element.modal();

    }

    var create_help_dialog = function (base) {

        var HELP_IMAGE = "plotter_scripts/EventPlot/EventPlot_help.jpg"

        var element = $('<div/>');

        // The documentation
        var doc = $('<div/>')
            .addClass('alert alert-info');

        doc.append(
            'EventPlot is a multi-lane timeline plot ' +
            'which supports interative zooming and timing calculation'
        );

        element.append(doc);

        var zoom = $("<div/>")
            .addClass("media-left");

        var addLabel = function (txt, cls) {
            return '<span class="label label-' + cls + '" + ">' +
                txt + '</span>'
        }

        var addListItem = function (txt) {
            return '<li class="list-group-item">' + txt +
                '</li>'
        }

        var addPlus = function () {
            return " + "
        }

        var addBadge = function (txt) {
            return '<span class="label label-default" style="border-radius: 10px">' +
                txt + '</span>'
        }

        element.append(
            '<img style="width: 100%;" class="media-object" src="' + base +
            HELP_IMAGE + '"/>'
        )

        element.append('<ul class="list-group">')
        element.append(addListItem('Scroll in the main area ' +
            addBadge("1") + " to zoom interactively"))
        element.append(addListItem(
            'Click and drag in the main area ' + addBadge(
                "1") + " to pan the zoom"))
        element.append(addListItem(
            'The summary of the plot is shown in ' +
            addBadge("2")))
        element.append(addListItem('Adjust the size of window ' +
            addBadge("4") +
            " set the X-Limits of the chart"))

        element.append(addListItem(addLabel("Right-Click",
                "default") +
            " to place marker " + addLabel("A", "success")))

        element.append(addListItem(addLabel("Ctrl", "primary") +
            " + " + addLabel("Right-Click", "default") +
            " to place marker " + addLabel("B", "danger")))

        element.append(addListItem(
            "The marker positions and delta will be shown in " +
            addBadge("3")))

        element.append(
            addListItem(addLabel("Ctrl", "primary") + addPlus() +
                addLabel("Alt", "primary") + addPlus() +
                addLabel("Right-Click", "default") +
                " on the rectange (eg. " + addBadge("6") +
                " ) to show info in " + addBadge("5")))

        element.append('</ul>')

        var dialog = create_dialog_body(element, "Help: EventPlot");
        dialog.show();

    }

    return {
        generate: generate,
        create_help_dialog: create_help_dialog
    };

}());
