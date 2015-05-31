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

var EventPlot = (function() {
    var margin, brush, x, ext, y1, y2, chart, main, xAxis, x1Axis,
        itemRects, items, clr, tip, lanes;

    var generate = function(div_name) {

        var json_file = "/static/plotter_data/" + div_name +
            ".json"

        $.getJSON(json_file, function(d) {

            items = d.data;
            lanes = d.lanes;
            var procs = d.keys;


            margin = {
                    top: 20,
                    right: 15,
                    bottom: 15,
                    left: 70
                }, width = 960 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom,
                miniHeight = lanes.length * 12 + 50,
                mainHeight = height - miniHeight - 50;

            x = d3.scale.linear()
                .domain([d3.min(items, function(d) {
                        return d.start
                    }),
                    d3.max(items, function(d) {
                        return d.end;
                    })
                ])
                .range([0, width]);

            clr_cat = d3.scale.category20();
            clr = d3.scale.ordinal()
                .range(clr_cat.range())
                .domain(procs);

            x1 = d3.scale.linear().range([0, width]);
            ext = d3.extent(lanes, function(d) {
                return d.id;
            });
            y1 = d3.scale.linear().domain([ext[0], ext[1] +
                1
            ]).range([0, mainHeight]);
            y2 = d3.scale.linear().domain([ext[0], ext[1] +
                1
            ]).range([0, miniHeight]);

            chart = d3.select('#' + div_name)
                .append('svg:svg')
                .attr('width', width + margin.right +
                    margin.left)
                .attr('height', height + margin.top +
                    margin.bottom)
                .attr('class', 'chart');

            chart.append('defs').append('clipPath')
                .attr('id', 'clip')
                .append('rect')
                .attr('width', width)
                .attr('height', mainHeight);

            main = chart.append('g')
                .attr('transform', 'translate(' + margin.left +
                    ',' + margin.top + ')')
                .attr('width', width)
                .attr('height', mainHeight)
                .attr('class', 'main');

            mini = chart.append('g')
                .attr('transform', 'translate(' + margin.left +
                    ',' + (mainHeight + 60) + ')')
                .attr('width', width)
                .attr('height', miniHeight)
                .attr('class', 'mini');

            main.append('g').selectAll('.laneLines')
                .data(lanes)
                .enter().append('line')
                .attr('x1', 0)
                .attr('y1', function(d) {
                    return d3.round(y1(d.id)) + 0.5;
                })
                .attr('x2', width)
                .attr('y2', function(d) {
                    return d3.round(y1(d.id)) + 0.5;
                })
                .attr('stroke', function(d) {
                    return d.label === '' ? 'white' :
                        'lightgray'
                });

            main.append('g').selectAll('.laneText')
                .data(lanes)
                .enter().append('text')
                .attr('x', -10)
                .text(function(d) {
                    return d.label;
                })
                .attr('y', function(d) {
                    return y1(d.id + .5);
                })
                .attr('dy', '0.5ex')
                .attr('text-anchor', 'end')
                .attr('class', 'laneText');

            mini.append('g').selectAll('.laneLines')
                .data(lanes)
                .enter().append('line')
                .attr('x1', 0)
                .attr('y1', function(d) {
                    return d3.round(y2(d.id)) + 0.5;
                })
                .attr('x2', width)
                .attr('y2', function(d) {
                    return d3.round(y2(d.id)) + 0.5;
                })
                .attr('stroke', function(d) {
                    return d.label === '' ? 'white' :
                        'lightgray'
                });


            mini.append('g').selectAll('.laneText')
                .data(lanes)
                .enter().append('text')
                .text(function(d) {
                    return d.label;
                })
                .attr('x', -10)
                .attr('y', function(d) {
                    return y2(d.id + .5);
                })
                .attr('dy', '0.5ex')
                .attr('text-anchor', 'end')
                .attr('class', 'laneText');

            xAxis = d3.svg.axis()
                .scale(x)
                .orient('bottom');

            x1Axis = d3.svg.axis()
                .scale(x1)
                .orient('bottom');

            tip = d3.tip()
                .attr('class', 'd3-tip')
                .offset([-10, 0])
                .html(function(d) {
                    return "<span style='color:white'>" +
                        d.name + "</span>";
                })

            main.append('g')
                .attr('transform', 'translate(0,' +
                    mainHeight + ')')
                .attr('class', 'main axis')
                .call(x1Axis);

            mini.append('g')
                .attr('transform', 'translate(0,' +
                    miniHeight + ')')
                .attr('class', 'axis')
                .call(xAxis);

            itemRects = main.append('g')
                .attr('clip-path', 'url(#clip)')

            mini.append('g').selectAll('miniItems')
                .data(getPaths(items))
                .enter().append('path')
                .attr('class', function(d) {
                    return 'miniItem'
                })
                .attr('d', function(d) {
                    return d.path;
                })
                .attr("stroke", function(d) {
                    return d.color
                })

            brush = d3.svg.brush()
                .x(x)
                .extent(x.domain())
                .on("brush", display);

            mini.append('g')
                .attr('class', 'brush')
                .call(brush)
                .selectAll('rect')
                .attr('y', 1)
                .attr('height', miniHeight - 1);

            display();

        });
    };


    var display = function() {
        var rects, labels, minExtent = brush.extent()[0],
            maxExtent = brush.extent()[1],
            visItems = items.filter(function(d) {
                return d.start < maxExtent && d.end > minExtent
            });

        main.call(tip);
        mini.select('.brush').call(brush.extent([minExtent,
            maxExtent
        ]));
        x1.domain([minExtent, maxExtent]);
        main.select('.main.axis').call(x1Axis);

        rects = itemRects.selectAll('rect')
            .data(visItems, function(d) {
                return d.id;
            })
            .attr('x', function(d) {
                return x1(d.start);
            })
            .attr('width', function(d) {
                return x1(d.end) - x1(d.start);
            })
            .attr("stroke", function(d) {
                return clr(d.name)
            })
            .attr("fill", function(d) {
                return clr(d.name)
            })
            .on("mouseover", tip.show)
            .on('mouseout', tip.hide)
            .on('mousemove', function() {

                tip.style("left", Math.max(0, d3.event.pageX -
                        60) + "px")
                    .style("top", (d3.event.pageY - 50) + "px");

            });

        rects.enter().append('rect')
            .attr('x', function(d) {
                return x1(d.start);
            })
            .attr('y', function(d) {
                return y1(d.lane) + .1 * y1(1) + 0.5;
            })
            .attr('width', function(d) {
                return x1(d.end) - x1(d.start);
            })
            .attr('height', function(d) {
                return .8 * y1(1);
            })
            .attr('class', function(d) {
                return 'mainItem'
            })
            .attr("stroke", function(d) {
                return clr(d.name)
            })
            .attr("fill", function(d) {
                return clr(d.name)
            })
            .on("mouseover", tip.show)
            .on('mouseout', tip.hide)
            .on('mousemove', function() {
                tip
                    .style("left", Math.max(0, d3.event.pageX -
                        60) + "px")
                    .style("top", (d3.event.pageY - 50) + "px");
            });

        rects.exit().remove();

    };

    var getPaths = function(items) {

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
        generate: generate
    };
}());
