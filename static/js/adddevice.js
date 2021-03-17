$(function () {
    var index = 0
    $(".confirmadd").click(function () {
        index = layer.open({
            type: 1,
            title: "确认删除",
            area: ["300px", "200px"],
            content: $('#confirmdel')
        })
    })
    $(".confirm").click(function () {
        var devicename = $("#devicenametlink").val()
        var protocoltype = $(".protocol").find("option:selected").text();
        protocoltype = protocoltype.toLowerCase()
        var delay = $("#delay").val()
        var length = $(".sensorname").length
        json = "{\"lng\": \"113.956591\",\"linkType\": \"" + protocoltype + "\",\"timescale\": " + delay + ",\"userId\": 200000152,\"deviceName\": \"" + devicename + "\",\"lat\": \"22.601376\",\"sensorList\": ["
        if (!devicename) {
            layer.close(index)
            alert("设备名称不能为空!")
        } else if (!delay) {
            layer.close(index)
            alert("延时不能为空!")
        } else {
            if (length == 0) {
                alert("请至少添加一个传感器!")
                layer.close(index)
            } else {
                layer.close(index)
                for (var i = 0; i < length; i++) {
                    var unit = document.getElementsByClassName("sensorunit")[i].value;
                    var sensorname = document.getElementsByClassName("sensorname")[i].value;
                    var sensortype = $(".sensortype").get(i).selectedIndex + 1
                    var order = document.getElementsByClassName("sensororder")[i].value;
                    var sensornum = $(".sensornum").get(i).selectedIndex
                    json += "{\"unit\": \"" + unit + "\",\"sensorName\":\"" + sensorname + "\",\"sensorType\": " + sensortype + ",\"ordernum\": " + order + ",\"decimal\": " + sensornum + " },"
                }
                json += "]}";
                $.ajax({
                    url: '/confirmadddevice',
                    type: 'GET',
                    data: {'json': json},
                    success: function (result) {
                        if (result == "00") {
                            alert("添加成功");
                            window.location.href = "/manageonline"
                        } else {
                            alert("添加失败，请刷新后重试");
                            window.location.href = "/manageonline"
                        }
                    },
                    error: function (XMLHttpResponse, textStatus, errorThrown) {
                        alert("添加失败，请刷新后重试!设备数不能超过20个!");
                        window.location.href = "/manageonline"
                        console.log(textStatus)
                        console.log(errorThrown)
                        console.log("error");
                    }
                })
            }
        }
    })
    $(".off").click(function () {
        layer.close(index)
    })
})