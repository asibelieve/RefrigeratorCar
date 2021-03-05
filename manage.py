from flask import Blueprint, render_template, request

from getDataAPI.api import getinfo

app = Blueprint("app_manage", __name__)


@app.route("/processmanage")
def processmanage():
    return render_template("DeviceManage.html")


@app.route("/manageonline")
def manageonline():
    '''json为接口的参数，具体参数说明见tlink，url为请求的路径，用request进行请求，输出r.text即为结果
            {//已隐藏相关信息,请自行请求获取
                "userId": "20******080",//用户Id必选参数
                "groupId":17, //设备分组条件 可选参数
                "isDelete":0,//设备状态 0 未删除 1已删除 2已禁用 可选参数，默认查询所有的设备
                "isLine":1,//设备在线状态 0 不在线，1在线 ，可选参数，默认查询在线和不在线的数据
                "isAlarms":"0",//设备是否报警，0 未报警，1已报警，可选参数，默认查询报警和未报警数据
                "currPage":1,//当前页码，必选参数，默认1 即第一页
                "pageSize":10//每页返回的数据条数，可选参数,默认返回10条，最大设置不能超过100条
            }
            return 的参数说明，result表示设备信息，length表示设备的个数，line表示当前状态是在线还是不在线
        '''
    json = "{\"userId\": \"200017377\", \"isDelete\": 0, \"isLine\": 1,\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getDevices'
    s = getinfo(json, url)
    s = s.replace('null', '0')
    try:
        res = eval(s)
        if res["dataList"]:
            return render_template("DeviceManage.html", result=res["dataList"], length=1, line=1)
    except:
        # print("暂时无数据")
        return render_template("DeviceManage.html", length=0, line=1)


'''动态监测的离线请求'''


@app.route('/manageoffline')
def manageoffline():
    '''json为接口的参数，具体参数说明见tlink，url为请求的路径，用request进行请求，输出r.text即为结果
        return 的参数说明，result表示设备信息，length表示设备的个数，line表示当前状态是在线还是不在线
    '''
    json = "{\"userId\": \"200017377\", \"isDelete\": 0, \"isLine\": 0,\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getDevices'
    s = getinfo(json, url)
    s = s.replace('null', '0')
    try:
        res = eval(s)
        if res["dataList"]:
            return render_template("DeviceManage.html", result=res["dataList"], length=1, line=0)
    except:
        # print("暂时无数据")
        return render_template("DeviceManage.html", length=0, line=0)


@app.route("/adddevicetlink")
def adddevicetlink():
    return render_template("adddevice.html")


@app.route("/updateDevicetlink")
def updateDevicetlink():
    deviceid = request.args.get("deviceid")
    json = "{\"userId\": \"200017377\", \"deviceId\": " + deviceid + ",\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getSingleDeviceDatas'
    s = getinfo(json, url)
    s = s.replace("null", "0")
    print(s)
    try:
        res = eval(s)
    except:
        print("error")
    return render_template("updateDevice.html", device=res["device"], deviceid=deviceid)


@app.route("/confirmupdate")
def confirmupdate():
    json = request.args.get("json")
    url = "http://api.tlink.io/api/device/updateDevice"
    json = json.encode("utf-8")
    s = getinfo(json, url)
    s = s.replace("null", "0")
    flag = ""
    try:
        s = eval(s)
        flag = s["flag"]
    except:
        flag = "01"
    return flag


@app.route("/confirmadddevice")
def confirmadddevice():
    json = request.args.get("json")
    print(json)
    url = "http://api.tlink.io/api/device/addDevice"
    json = json.encode("utf-8")
    s = getinfo(json, url)
    print(s)
    s = s.replace("null", "0")
    flag = ""
    try:
        s = eval(s)
        flag = s["flag"]
    except:
        flag = "01"
    return flag


@app.route("/deldevicetlink")
def deldevice():
    deviceid = request.args.get("deviceid")
    json = "{\"userId\": \"200017377\", \"deviceId\": " + deviceid + " }"
    print(json)
    url = "http://api.tlink.io/api/device/deleteDevice"
    s = getinfo(json, url)
    s = s.replace("null", "0")
    flag = ""
    try:
        s = eval(s)
        flag = s["flag"]
    except:
        flag = "01"
    return flag


@app.route("/delsensorbyid")
def delsensor():
    json = request.args.get("json")
    url = "http://api.tlink.io/api/device/updateDevice"
    json = json.encode("utf-8")
    s = getinfo(json, url)
    s = s.replace("null", "0")
    flag = ""
    try:
        s = eval(s)
        flag = s["flag"]
    except:
        flag = "01"
    return flag
