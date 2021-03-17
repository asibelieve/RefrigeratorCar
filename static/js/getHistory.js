var settings = {
    "url": "http://api.tlink.io/api/device/getSensorHistroy",
    "method": "GET",
    "timeout": 0,
    "headers": {
        "tlinkAppId": "36b4943878cb4f8eab2c6473f0169917",
        "Authorization": "65062e4c-ff63-4c53-8021-5217d4fe6307",
        "Content-Type": "text/plain"
    },
    "data": "{" +
        "\r\n  \"userId\":200000152," +
        "\r\n  \"deviceId\":200007683," +
        "\r\n  \"sensorId\":200070123," +
        "\r\n  \"startDate\" : \"2020-01-10 10:01:11\"," +
        "\r\n  \"endDate\" : \"2020-01-10 11:01:11\"," +
        "\r\n  \"currPage\":1," +
        "\r\n  \"pageSize\":10" +
        "\r\n}",
};

$.ajax(settings).done(function (response) {
    console.log(response);
});