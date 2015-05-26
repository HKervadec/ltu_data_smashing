$(document).ready(
    $(function () {
        $('#signalsc').highcharts({
            title: {
                text: 'Source signals',
               
            },
            yAxis: {
                title: {
                    text: 'Value'
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
            series: signals,
            plotOptions:{
                series:{
                    lineWidth: 1
                }
            },
        });
    })
)