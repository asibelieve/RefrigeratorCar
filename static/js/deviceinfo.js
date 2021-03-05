$(function () {
    $(".tlinkPolice").click(function () {
        var id = this.id;
        $.ajax({
            url: '/alarmrecord',
            type: 'GET',
            data: {'id': id},
            success: function (data) {
                console.log(data);
                if (data == "0") {
                    $(".noalarmdata")[0].style.display = "block"
                    $(".alarmtable")[0].style.display = "none"
                } else {
                    data = eval(data);
                    $.each(data, function (i, result) {
                        var item = "<tr>";
                        item = item + "<td>" + result["sensorName"] + "</td>";
                        item = item + "<td>" + result["sensorId"] + "</td>";
                        item = item + "<td>" + result["status"] + "</td>";
                        item = item + "<td>" + result["triggerDate"] + "</td>";
                        item = item + "<td>" + result["triggerContent"] + "</td>";
                        item = item + "<td>" + result["triggerVal"] + "</td>";
                        item = item + "</tr>"
                        $(".alarm")[0].append(item);
                    });
                    $(".alarm")[0].style.display = "block"
                }
            }
        })
        layer.open({
            type: 1,
            title: "报警记录",
            area: ["400px", "400px"],
            content: $('#alarmrecord')
        });
    })
})