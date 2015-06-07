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

        var margin, brush, x, ext, y1,summarychart, main, xAxis,
            x1Axis,
            itemRects, items, clr, tip, lanes;

        var json_file = "/static/plotter_data/" + div_name +
            ".json"

        $.getJSON(json_file, function (d) {

            items = d.data;
            lanes = d.lanes;
            var procs = d.keys;

            margin = {
                    top: 20,
                    right: 15,
                    bottom: 15,
                    left: 70
                }, width = 960 - margin.left - margin.right,

            mainHeight = 300 - margin.top - margin.bottom;
            miniHeight = lanes.length * 12 + 50,

            x = d3.scale.linear()
                .domain([d3.min(items, function (d) {
                        return d.start
                    }),
                    d3.max(items, function (d) {
                        return d.end;
                    })
                ])
                .range([0, width]);

            var zoom_scale = d3.scale.linear()
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
            clr_cat = d3.scale.category20();
            clr = d3.scale.ordinal()
                .range(clr_cat.range())
                .domain(procs);

            x1 = d3.scale.linear()
                .range([0, width]);
            ext = d3.extent(lanes, function (d) {
                return d.id;
            });
            y1 = d3.scale.linear()
                .domain([ext[0], ext[1] +
                    1
                ])
                .range([0, mainHeight]);
            y2 = d3.scale.linear()
                .domain([ext[0], ext[1] +
                    1
                ])
                .range([0, miniHeight]);

            xAxis = d3.svg.axis()
                .scale(x)
                .orient('bottom');

            var ePlot;


            chart = d3.select('#' + div_name)
                .append('svg:svg')
                .attr('width', width + margin.right +
                    margin.left)
                .attr('height', mainHeight + margin.top + margin.bottom + 5)
                .attr('class', 'chart')

            var summary = d3.select("#" + div_name).append("svg:svg")
                .attr('width', width + margin.right +
                    margin.left)
                .attr('height', miniHeight + margin.bottom + margin.top + 5)
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
                    return d3.round(y1(d.id)) + 0.5;
                })
                .attr('x2', width)
                .attr('y2', function (d) {
                    return d3.round(y1(d.id)) + 0.5;
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
                    return y1(d.id + .5);
                })
                .attr('dy', '0.5ex')
                .attr('text-anchor', 'end')
                .attr('class', 'laneText');

            x1Axis = d3.svg.axis()
                .scale(x1)
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
                .call(x1Axis);

            itemRects = main.append('g')
                .attr('clip-path', 'url(#clip)')

            var ePlot;

            ePlot = {
                margin: margin,
                chart: chart,
                summary: summary,
                mainHeight: mainHeight,
                miniHeight: miniHeight,
                width: width,
                x: x,
                x1: x1,
                ext: ext,
                y1: y1,
                y2: y2,
                main: main,
                xAxis: xAxis,
                x1Axis: x1Axis,
                itemRects: itemRects,
                items: items,
                clr: clr,
                tip: tip,
                lanes: lanes,
            };

            ePlot.mini = createMini(ePlot);
            var zoomed = function () {

                if (zoom_scale.domain()[0] < xMin) {
                    zoom.translate([zoom.translate()[
                            0] - zoom_scale(xMin) +
                        zoom_scale.range()[0], zoom.translate()[
                            1]
                    ]);
                } else if (zoom_scale.domain()[1] > xMax) {
                    zoom.translate([zoom.translate()[
                            0] - zoom_scale(xMax) +
                        zoom_scale.range()[1], zoom.translate()[
                            1]
                    ]);

                }


                brush.extent(zoom_scale.domain());
                ePlot.mini.select(".brush").call(brush);

                updateGraph(ePlot, zoom_scale.domain()[0], zoom_scale
                    .domain()[1]);

            };

            var _brushed_event = function () {
                var brush_xmin = brush.extent()[0];
                var brush_xmax = brush.extent()[1];

                var t = zoom.translate(),
                    new_domain = brush.extent(),
                    scale;

                  /*
                   * scale = x.range()[1] - x.range[0]
                            --------------------------
                           x(x.domain()[1] - x.domain()[0])

                                               _                                   _
                    new_domain[0] =  x.invert | x.range()[0]  -   z.translate()[0]  |
                                              |                 ------------------- |
                                              |_                     z.scale()     _|



                  translate[0] = x.range()[0] - x(new_domain[0])) * zoom.scale()
                 */

                  scale = (width) / x(new_domain[1] - new_domain[0]);
                  zoom.scale(scale);
                  t[0] = x.range()[0] - (x(new_domain[0]) * scale);
                  zoom.translate(t);


                  updateGraph(ePlot, brush.extent()[0],
                  brush.extent()[1]);

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
                    .attr('height', miniHeight - 1);

                var zoom = d3.behavior.zoom()
                    .x(zoom_scale)
                    .on(
                        "zoom", zoomed)
                    .scaleExtent([1, 32]);
                chart.call(zoom);

            updateGraph(ePlot, xMin, xMax);
            return ePlot;

        });
    };

    var createMini = function (ePlot) {

        var mini = ePlot.summary.append('g')
            .attr("transform", "translate(" + ePlot.margin.left + "," + ePlot.margin.top + ")")
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
                return d3.round(ePlot.y2(d.id)) + 0.5;
            })
            .attr('x2', ePlot.width)
            .attr('y2', function (d) {
                return d3.round(ePlot.y2(d.id)) + 0.5;
            })
            .attr('stroke', function (d) {
                return d.label === '' ? 'white' :
                    'lightgray'
            });

        mini.append('g')
            .attr('transform', 'translate(0,' +
                ePlot.miniHeight + ')')
            .attr('class', 'axis')
            .call(ePlot.xAxis);


        mini.append('g')
            .selectAll('miniItems')
            .data(getPaths(ePlot.items, ePlot.x, ePlot.y2, ePlot.clr))
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
                return ePlot.y2(d.id + .5);
            })
            .attr('dy', '0.5ex')
            .attr('text-anchor', 'end')
            .attr('class', 'laneText');

        return mini;
    };


    var updateGraph = function (ePlot, xMin, xMax) {


        var rects, labels;
        var visItems = ePlot.items.filter(function (d) {
            return d.start < xMax && d.end > xMin
        });

        ePlot.main.call(ePlot.tip);
        ePlot.x1.domain([xMin, xMax]);
        ePlot.main.select('.main.axis')
            .call(ePlot.x1Axis);

        rects = ePlot.itemRects.selectAll('rect')
            .data(visItems, function (d) {
                return d.id;
            })
            .attr('x', function (d) {
                return ePlot.x1(d.start);
            })
            .attr('width', function (d) {
                return ePlot.x1(d.end) - ePlot.x1(d.start);
            })
            .attr("stroke", function (d) {
                return ePlot.clr(d.name)
            })
            .attr("fill", function (d) {
                return ePlot.clr(d.name)
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
            .attr('x', function (d) {
                return ePlot.x1(d.start);
            })
            .attr('y', function (d) {
                return ePlot.y1(d.lane) + .1 * ePlot.y1(1) +
                    0.5;
            })
            .attr('width', function (d) {
                return ePlot.x1(d.end) - ePlot.x1(d.start);
            })
            .attr('height', function (d) {
                return .8 * ePlot.y1(1);
            })
            .attr('class', function (d) {
                return 'mainItem'
            })
            .attr("stroke", function (d) {
                return ePlot.clr(d.name)
            })
            .attr("fill", function (d) {
                return ePlot.clr(d.name)
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

    var getPaths = function (items, x, y2, clr) {

        var paths = {},
            d, offset = .5 * y2(1) + 0.5,
            result = [];
        for (var i = 0; i < items.length; i++) {
            d = items[i];
            if (!paths[d.name]) paths[d.name] = '';
            paths[d.name] += ['M', x(d.start), (y2(d.lane) + offset),
                'H', x(d.end)
            ].join(' ');
        }


        for (var name in paths) {
            result.push({
                color: clr(name),
                path: paths[name]
            });
        }

        return result;

    }

    return {
        generate: generate,
    };

}());
