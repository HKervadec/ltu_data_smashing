$(document).ready(
    $(function () {
        $('#independetc').highcharts({
            title: {
                text: 'Independet signal copies',
                //x: -20 //center
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
            series: independent_stream_copies,
            plotOptions:{
                series:{
                    lineWidth: 1
                }
            },
        });
    })
)