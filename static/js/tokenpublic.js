$(function () {
    storage = window.sessionStorage;
    var token = storage.getItem("token")
    var key = storage.getItem("key")
    console.log("i am key:" + key)
    console.log("i am token:" + token)
})