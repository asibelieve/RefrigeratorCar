res = dynamicdata.data
$(function () {
    var index = 0;
    $(".assessdynamicshow").click(function () {
        index = layer.open({
            type: 1,
            title: "品质变化",
            area: ["600px", "500px"],
            content: $('#visualdiv')
        })
        var ec1 = echarts.init(document.getElementById("visualdiv"))
        ec1.showLoading()
        if (res.length == 0) {
            alert("没有已评测过的数据!")
            layer.close(index)
        } else {
            var fresh = []
            var picname = []
            var time = []
            for (var i = 0, j = res.length; i < j; i++) {
                fresh.push(res[i][8])
                picname.push(res[i][2])
                time.push(res[i][5])
            }
            option = {
                title: {
                    text: '品质动态化监测',
                    textStyle: {
                        align: 'center',
                        fontSize: 20,
                    },
                    top: '5%',
                    left: 'center',
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: '图片名称:{c1}<br>创建时间:{b0}<br>新鲜度: {c0}'
                },
                grid: {
                    left: '0%',
                    right: '3%',
                    bottom: '3%',
                    containLabel: true
                },
                toolbox: {
                    feature: {
                        saveAsImage: {}
                    }
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data: time,
                    axisLine: {
                        show: true
                    },
                },
                yAxis: {
                    type: 'category',
                    inverse: true,
                    axisLine: {
                        show: true
                    },
                },
                //坐标轴伸缩，可拖动，可滚轮
                dataZoom: [
                    {
                        type: 'slider',
                        show: true,
                        xAxisIndex: [0],
                        start: 40,
                        end: 100
                    },
                    {
                        type: 'inside',
                        show: true,
                        xAxisIndex: [0],
                        start: 40,
                        end: 100
                    }
                ],
                series: [{
                    name: "时间",
                    type: 'line',
                    data: fresh,
                    showAllSymbol: true,
                    // symbol: 'image://./static/images/guang-circle.png',
                    symbol: 'circle',
                    symbolSize: 10,
                    itemStyle: {
                        color: "#6c50f3",
                        borderColor: "#fff",
                        borderWidth: 3,
                        shadowColor: 'rgba(0, 0, 0, .3)',
                        shadowBlur: 0,
                        shadowOffsetY: 2,
                        shadowOffsetX: 2,
                    },
                    label: {
                        show: true,
                        position: 'top',
                        textStyle: {
                            color: '#6c50f3',
                        }
                    },
                    lineStyle: {
                        type: 'solid',
                        width: 3,
                        shadowColor: 'rgba(0, 0, 0, 0.5)',
                        shadowBlur: 6,
                        shadowOffsetY: 6,
                    }
                }, {
                    name: "时间",
                    type: 'line',
                    data: picname,
                    showAllSymbol: true,
                    // symbol: 'image://./static/images/guang-circle.png',
                    symbol: 'circle',
                    symbolSize: 10,
                    label: {
                        show: true,
                        position: 'top',
                        textStyle: {
                            color: '#6c50f3',
                        }
                    },
                    itemStyle: {
                        color: "#00ca95",
                        borderColor: "#fff",
                        borderWidth: 3,
                        shadowColor: 'rgba(0, 0, 0, .3)',
                        shadowBlur: 0,
                        shadowOffsetY: 2,
                        shadowOffsetX: 2,
                    },
                    lineStyle: {
                        type: 'solid',
                        width: 3,
                        shadowColor: 'rgba(0, 0, 0, 0.5)',
                        shadowBlur: 6,
                        shadowOffsetY: 6,
                    }
                },]
            }
            ec1.hideLoading()
            ec1.setOption(option)
        }
    })
})