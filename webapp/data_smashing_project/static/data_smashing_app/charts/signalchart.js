$(document).ready(
    $(function () {
        $('#hccontainer').highcharts({
            title: {
                text: 'Source signal',
                x: -20 //center
            },
            yAxis: {
                title: {
                    text: 'Signal'
                },
                plotLines: [{
                    value: 0,
                    width: 0.5,
                    color: '#808080'
                }]
            },
            xAxis: {
                title: {
                    enabled: false
                }
            },
            series: [{
                showInLegend: false,
                data: signal
            }],
            plotOptions:{
                series:{
                    lineWidth: 1
                }
            },
        });
    })
)