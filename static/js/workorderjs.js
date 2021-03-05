/**
 *
 */
function getCookieValue(name) {
    /**获取cookie的值，根据cookie的键获取值**/
        //用处理字符串的方式查找到key对应value
    var name = escape(name);
    //读cookie属性，这将返回文档的所有cookie
    var allcookies = document.cookie;
    //查找名为name的cookie的开始位置
    name += "=";
    var pos = allcookies.indexOf(name);
    //如果找到了具有该名字的cookie，那么提取并使用它的值
    if (pos != -1) {                       //如果pos值为-1则说明搜索"version="失败
        var start = pos + name.length;         //cookie值开始的位置
        var end = allcookies.indexOf(";", start);    //从cookie值开始的位置起搜索第一个";"的位置,即cookie值结尾的位置
        if (end == -1) end = allcookies.length;    //如果end值为-1说明cookie列表里只有一个cookie
        var value = allcookies.substring(start, end); //提取cookie的值
        return (value);              //对它解码
    } else { //搜索失败，返回空字符串
        return "";
    }
}

window.onload = function () {
    var loginid = getCookieValue("loginid");
    if (loginid == "") {
        window.location.href = "http://localhost:8080/FileManage/login";
    } else {
        $(".qc-user-name").html(loginid);
        $(".qc-user-name").css("title", loginid);
    }
    $.ajax({
        type: "GET",
        url: "http://localhost:8080/FileManage/api/getuserinfo?",
        dataType: "json",
        data: {
            loginid: loginid,
            token: sessionStorage.getItem("token")
        },
        async: false,
        success: function (result) {

            $.each(result.data.list, function (i, val) {
                var list = "<li class=\"J-qcTopNavSearchItem menu-search-select\"><a href=\"#\" data-event=\"nav\">" + val["title"] + "</a></li>";
                $("#productService > div > div.menu-list-search.J-headerProductSearchBar > ul").append(list);
            })
        },
        error: function (result) {
            console.log("error: " + JSON.stringify(result));
        }
    });
}