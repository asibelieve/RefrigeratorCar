$(function () {
    storage = window.sessionStorage;
    var operator = nonedata.operator;
    if (operator == "quit") {
        storage.removeItem("token")
        storage.removeItem("key")
    } else {
        var token = storage.getItem("token")
        var key = storage.getItem("key")
        console.log("i am key:" + key)
        console.log("i am token:" + token)
        if (token == null && key == null) {
            console.log("i am here1")
            token = nonedata.data;
            key = nonedata.key;
            window.sessionStorage.setItem("token", token);
            window.sessionStorage.setItem("key", key)
        } else if (token == "" && key == "") {
            console.log("i am here1")
            token = nonedata.data;
            key = nonedata.key;
            window.sessionStorage.setItem("token", token);
            window.sessionStorage.setItem("key", key)
        } else {
            console.log("i am here3")
            token = storage.getItem("token")
            key = storage.getItem("key")
        }
        console.log("i am key:" + key)
        console.log("i am token:" + token)
    }
})