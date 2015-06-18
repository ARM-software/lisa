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
 * File:        ILinePlot.js
 * ----------------------------------------------------------------
 * $
 */
var ILinePlot = ( function() {
   var convertToDataTable = function (d, index_col) {

        var columns = _.keys(d);
        var out = [];
        var index_col_default = false;
        var index;

        if (index_col == undefined) {

            var index = [];

            columns.forEach(function(col) {
                index = index.concat(Object.keys(d[col]));
            });

            index = $.unique(index);
            index_col_default = true;
            index = index.sort(function(a, b) {
                return (parseFloat(a) - parseFloat(b));
            });
        } else {
            index = d[index_col];
            columns.splice(columns.indexOf(index_col), 1);
        }

        for (var ix in index) {

            var ix_val = ix;

            if (index_col_default)
                ix_val = index[ix];

            var row = [parseFloat(ix_val)];
            columns.forEach(function(col) {

                var val = d[col][ix_val];
                if (val == undefined)
                    val = null;

                row.push(val);
            });
            out.push(row);
        }

        var labels = ["index"].concat(columns);
        return {
            data: out,
            labels: labels
        }
    };

    var generate = function(div_name) {
        var json_file = "/static/plotter_data/" + div_name + ".json";
            $.getJSON( json_file, function( data ) {
                create_graph(data);
            });
    };

    var create_graph = function(t_info) {
        var tabular = convertToDataTable(t_info.data, t_info.index_col);

        new Dygraph(document.getElementById(t_info.name), tabular.data, {
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
            valueRange: t_info.valueRange

        })
    };

    return {
        generate: generate
    };

}());
