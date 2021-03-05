var data = updatedata.data
var sensorid = ""
var index2 = 0

function confirmid(id) {
    sensorid = id
    index2 = layer.open({
        type: 1,
        title: "确认删除",
        area: ["400px", "200px"],
        content: $('#confirmdelsensor')
    })
}

$(function () {
    var index = 0;
    $(".confirmupdate").click(function () {
        index = layer.open({
            type: 1,
            title: "确认删除",
            area: ["400px", "200px"],
            content: $('#confirmdel')
        })
    })
    $(".confirm").click(function () {
        var res = eval(data)
        var devicename = $("#devicenametlink").val()
        var protocoltype = $(".protocol").find("option:selected").text();
        var delay = $("#delay").val()
        var deviceid = res["id"]
        var json = "{\"lng\":" + res["lng"] + ",\"linkType\": \"" + protocoltype + "\",\"timescale\": " + delay + ",\"userId\": 200017377,\"deviceName\": \"" + devicename + "\",\"lat\": " + res["lat"] + "," +
            "\"deviceId\":" + res["id"] + ",\"sensorList\": ["
        var length = $(".sensorname").length
        var sensor = res["sensorsList"]
        for (var i = 0; i < length; i++) {
            var sensorname = document.getElementsByClassName("sensorname")[i].value;
            var sensortype = $(".sensortype").get(i).selectedIndex + 1
            var sensornum = $(".sensornum").get(i).selectedIndex
            var unit = document.getElementsByClassName("sensorunit")[i].value;
            var order = document.getElementsByClassName("sensororder")[i].value;
            json += "{\"sensorId\":" + sensor[i]["id"] + ",\"unit\": \"" + unit + "\",\"sensorName\":\"" + sensorname + "\",\"sensorType\": " + sensortype + ",\"ordernum\": " + order + ",\"decimal\": " + sensornum + " },"
        }
        json += "]}";
        $.ajax({
            url: '/confirmupdate',
            type: 'GET',
            data: {'json': json},
            success: function (result) {
                alert("修改成功");
                window.location.href = "/updateDevicetlink?deviceid=" + deviceid
            },
            error: function (XMLHttpResponse, textStatus, errorThrown) {
                alert("修改失败，请刷新后重试");
                window.location.href = "/updateDevicetlink?deviceid=" + deviceid
                console.log(textStatus)
                console.log(errorThrown)
                console.log("error");
            }
        })

        layer.close(index)
    })
    $(".off").click(function () {
        layer.close(index)
    })
    $(".confirmsensor").click(function () {
        var res = eval(data)
        var devicename = $("#devicenametlink").val()
        var protocoltype = $(".protocol").find("option:selected").text();
        var delay = $("#delay").val()
        var deviceid = res["id"]
        var json = "{\"lng\":" + res["lng"] + ",\"linkType\": \"" + protocoltype + "\",\"timescale\": " + delay + ",\"userId\": 200017377,\"deviceName\": \"" + devicename + "\",\"lat\": " + res["lat"] + "," +
            "\"deviceId\":" + res["id"] + ",\"delSensorIds\":" + sensorid + ",\"sensorList\": ["
        var length = res["sensorsList"].length
        var sensor = res["sensorsList"]
        for (var i = 0; i < length; i++) {
            var sensorname = document.getElementsByClassName("sensorname")[i].value;
            var sensortype = $(".sensortype").get(i).selectedIndex + 1
            var sensornum = $(".sensornum").get(i).selectedIndex
            var unit = document.getElementsByClassName("sensorunit")[i].value;
            var order = document.getElementsByClassName("sensororder")[i].value;
            json += "{\"sensorId\":" + sensor[i]["id"] + ",\"unit\": \"" + unit + "\",\"sensorName\":\"" + sensorname + "\",\"sensorType\": " + sensortype + ",\"ordernum\": " + order + ",\"decimal\": " + sensornum + " },"
        }
        json += "]}";
        $.ajax({
            url: '/confirmupdate',
            type: 'GET',
            data: {'json': json},
            success: function (result) {
                alert("删除成功");
                window.location.href = "/updateDevicetlink?deviceid=" + deviceid
            },
            error: function (XMLHttpResponse, textStatus, errorThrown) {
                alert("删除失败，请刷新后重试");
                window.location.href = "/updateDevicetlink?deviceid=" + deviceid
                console.log(textStatus)
                console.log(errorThrown)
                console.log("error");
            }
        })
        layer.close(index2)
    })
    $(".offsensor").click(function () {
        layer.close(index2)
    })

})