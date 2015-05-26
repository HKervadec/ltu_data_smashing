$(document).ready(
    $(function () {
        $('#sumationsc').highcharts({
            title: {
                text: 'Stream sumations',
                //x: -20 //center
            },
            yAxis: {
                title: {
                    text: 'Values'
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
            series: stream_sumations,
            plotOptions:{
                series:{
                    lineWidth: 1
                }
            },
        });
    })
)