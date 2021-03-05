$(function () {
    storage = window.sessionStorage;
    var token = storage.getItem("token")
    var key = storage.getItem("key")
    console.log("i am key:" + key)
    console.log("i am token:" + token)
    if (token == null && key == null) {
        window.location.href = "/wait"
    } else if (token == "" && key == "") {
        window.location.href = "/wait"
    } else {
        $.ajax({
            url: '/confirmtoken',
            type: 'GET',
            data: {'token': token, 'key': key},
            success: function (result) {
                if (result != "true") {
                    window.location.href = "/wait"
                }
            },
            error: function (XMLHttpResponse, textStatus, errorThrown) {
                window.location.href = "/wait"
                console.log(textStatus)
                console.log(errorThrown)
                console.log("error");
            }
        })
    }
})