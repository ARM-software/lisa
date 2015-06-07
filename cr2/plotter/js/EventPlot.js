/* $Copyright:
 * ----------------------------------------------------------------
 * This confidential and proprietary software may be used only as
 * authorised by a licensing agreement from ARM Limited
 *  (C) COPYRIGHT 2015 ARM Limited
 *       ALL RIGHTS RESERVED
 * The entire notice above must be reproduced on all authorised
 * copies and copies may only be made to the extent permitted
 * by a licensing agreement from ARM Limited.
 * ----------------------------------------------------------------
 * File:        EventPlot.js
 * ----------------------------------------------------------------
 * $
 */
var EventPlot = (function () {

    var generate = function (div_name) {

        var margin, brush, x, ext, yMain, chart, main,
            mainAxis,
            itemRects, items, colourAxis, tip, lanes;

        var json_file = "/static/plotter_data/" + div_name +
            ".json"

        $.getJSON(json_file, function (d) {

            items = d.data;
            lanes = d.lanes;
            var names = d.keys;
            var showSummary = d.showSummary;

            margin = {
                    top: 20,
                    right: 15,
                    bottom: 15,
                    left: 70
                }, width = 960 - margin.left - margin.right,

                mainHeight = 300 - margin.top - margin.bottom;

            x = d3.scale.linear()
                .domain([d3.min(items, function (d) {
                        return d.start
                    }),
                    d3.max(items, function (d) {
                        return d.end;
                    })
                ])
                .range([0, width]);

            var zoomScale = d3.scale.linear()
                .domain([d3.min(items, function (d) {
                        return d.start
                    }),
                    d3.max(items, function (d) {
                        return d.end;
                    })
                ])
                .range([0, width]);

            var xMin = x.domain()[0];
            var xMax = x.domain()[1];


            //Colour Ordinal scale. Uses Category20 Colors
            colours = d3.scale.category20();
            colourAxis = d3.scale.ordinal()
                .range(colours.range())
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


            chart = d3.select('#' + div_name)
                .append('svg:svg')
                .attr('width', width + margin.right +
                    margin.left)
                .attr('height', mainHeight + margin.top +
                    margin.bottom + 5)
                .attr('class', 'chart')


            chart.append('defs')
                .append('clipPath')
                .attr('id', 'clip')
                .append('rect')
                .attr('width', width)
                .attr('height', mainHeight);

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
                .attr('x', -10)
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
                .offset([-10, 0])
                .html(function (d) {
                    return "<span style='color:white'>" +
                        d.name + "</span>";
                })

            main.append('g')
                .attr('transform', 'translate(0,' +
                    mainHeight + ')')
                .attr('class', 'main axis')
                .call(mainAxis);

            itemRects = main.append('g')
                .attr('clip-path', 'url(#clip)')

            var ePlot;

            ePlot = {
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
                itemRects: itemRects,
                items: items,
                colourAxis: colourAxis,
                tip: tip,
                lanes: lanes,
            };

            if (showSummary)
                ePlot.mini = drawMini(ePlot);

            var transform = function (d) {}


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

                if (showSummary) {
                    brush.extent(zoomScale.domain());
                    ePlot.mini.select(".brush")
                        .call(
                            brush);
                }

                ePlot.itemRects.selectAll("rect")
                    .attr(
                        "transform",
                        function (d) {
                            return "translate(" + (
                                zoomScale(d.start)
                            ) + ",0 )";
                        })
                    .attr("width", function (d) {
                        return Math.max(zoomScale(
                                (zoomScale.domain()[
                                        0] + d.end -
                                    d.start)),
                            1)
                    })

                brushScale.domain(zoomScale.domain());
                ePlot.main.select('.main.axis')
                    .call(ePlot.mainAxis)
            };

            if (showSummary) {
                var _brushed_event = function () {
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

                    scale = (width) / x(x.domain()[0] + new_domain[1] -
                        new_domain[0]);
                    zoom.scale(scale);
                    t[0] = x.range()[0] - (x(new_domain[
                        0]) * scale);
                    zoom.translate(t);

                    brushScale.domain(brush.extent())
                    ePlot.itemRects.selectAll("rect")
                        .attr(
                            "transform",
                            function (d) {
                                return "translate(" + (
                                        brushScale(d.start)) +
                                    ",0 )";
                            })
                        .attr("width", function (d) {
                            return Math.max(brushScale((brushScale.domain()[
                                    0] +
                                d.end -
                                d.start
                            )), 1)
                        })

                    ePlot.main.select('.main.axis')
                        .call(ePlot.mainAxis)

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
                .scaleExtent([1, 32]);
            chart.call(zoom);

            drawMain(ePlot, xMin, xMax);
            return ePlot;

        });
    };

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
                ePlot.margin.top + 5)
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


        mini.append('g')
            .selectAll('miniItems')
            .data(getPaths(ePlot.items, ePlot.x, ePlot.yMini, ePlot.colourAxis))
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
        var visItems = ePlot.items.filter(function (d) {
            return d.start < xMax && d.end > xMin
        });

        ePlot.main.call(ePlot.tip);
        ePlot.brushScale.domain([xMin, xMax]);
        ePlot.main.select('.main.axis')
            .call(ePlot.mainAxis);

        rects = ePlot.itemRects.selectAll('rect')
            .attr("transform", function (d) {
                return "translate(" + ePlot.x(d.start) + ", 0)";
            })
            .attr('x', 0)
            .data(visItems, function (d) {
                return d.id;
            })
            .attr('width', function (d) {
                return Math.max(ePlot.brushScale(d.end) - ePlot.brushScale(d.start),
                    1);
            })
            .attr("fill", function (d) {
                return ePlot.colourAxis(d.name)
            })
            .on("mouseover", ePlot.tip.show)
            .on('mouseout', ePlot.tip.hide)
            .on('mousemove', function () {

                ePlot.tip.style("left", Math.max(0, d3.event.pageX -
                        60) + "px")
                    .style("top", (d3.event.pageY - 50) + "px");

            });

        rects.enter()
            .append('rect')
            .attr("transform", function (d) {
                return "translate(" + ePlot.x(d.start) + ", 0)";
            })
            .attr('x', 0)
            .attr('y', function (d) {
                return ePlot.yMain(d.lane) + .1 * ePlot.yMain(1) +
                    0.5;
            })
            .attr('width', function (d) {
                return Math.max(ePlot.brushScale(d.end) - ePlot.brushScale(d.start),
                    1);
            })
            .attr('height', function (d) {
                return 0.8 * ePlot.yMain(1);
            })
            .attr('class', function (d) {
                return 'mainItem'
            })
            .attr("fill", function (d) {
                return ePlot.colourAxis(d.name)
            })
            .on("mouseover", ePlot.tip.show)
            .on('mouseout', ePlot.tip.hide)
            .on('mousemove', function () {
                ePlot.tip
                    .style("left", Math.max(0, d3.event.pageX -
                        60) + "px")
                    .style("top", (d3.event.pageY - 50) + "px");
            });

        rects.exit()
            .remove();

    };

    var getPaths = function (items, x, yMini, colourAxis) {

        var paths = {},
            d, offset = 0.5 * yMini(1) + 0.5,
            result = [];
        for (var i = 0; i < items.length; i++) {
            d = items[i];

            var end = d.end;
            if (x(x.domain()[0] + d.end - d.start) < 1)
                end = x(d.start) + 1;
            else
                end = x(d.end);

            if (!paths[d.name]) paths[d.name] = '';
            paths[d.name] += ['M', x(d.start), (yMini(d.lane) + offset),
                'H', end
            ].join(' ');
        }


        for (var name in paths) {
            result.push({
                color: colourAxis(name),
                path: paths[name]
            });
        }

        return result;

    }

    return {
        generate: generate,
    };

}());
