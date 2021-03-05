var deviceid = ""

function confirmid(id) {
    deviceid = id
    console.log(deviceid)
    index = layer.open({
        type: 1,
        title: "确认删除",
        area: ["400px", "200px"],
        content: $('#confirmdel')
    })
}

$(function () {
    $(".confirm").click(function () {
        $.ajax({
            url: '/deldevicetlink',
            type: 'GET',
            data: {'deviceid': deviceid},
            success: function (result) {
                if (result == "00") {
                    alert("删除成功");
                    window.location.href = "/manageonline"
                } else {
                    alert("删除失败，请刷新后重试");
                    window.location.href = "/manageonline"
                }
            },
            error: function (XMLHttpResponse, textStatus, errorThrown) {
                alert("删除失败，请刷新后重试");
                window.location.href = "/manageonline"
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
})