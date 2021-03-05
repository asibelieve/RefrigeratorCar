import datetime
import os

import pymysql
from flask import render_template
from flask import request, Blueprint
from werkzeug.utils import secure_filename

from getDataAPI.api import getinfo
from getDataAPI.filemanage import removedir, getfilelist

project_path = "G:/Pycharm/Fish/static/uploadFileDir/"
result_path = "G:/Pycharm/Fish/static/result/"
app = Blueprint("app", __name__)

'''主页面'''


@app.route('/')
def index():
    token = "7070619e-e4af-44df-b213-1f21a7f01cc2"
    return render_template("index.html", token=token, key="", operator="login")


'''动态监测的在线请求'''


@app.route('/online')
def online():
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
            return render_template("DynamicListener.html", result=res["dataList"], length=1, line=1)
    except:
        # print("暂时无数据")
        return render_template("DynamicListener.html", length=0, line=1)


'''动态监测的离线请求'''


@app.route('/offline')
def offline():
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
            return render_template("DynamicListener.html", result=res["dataList"], length=1, line=0)
    except:
        # print("暂时无数据")
        return render_template("DynamicListener.html", length=0, line=0)


'''设备详情'''


@app.route('/deviceDetail')
def deviceDetail():
    '''
    {
        "userId": 1*****08,//用户Id必传参数
        "deviceId":200****42,//设备Id 可选参数 设备Id序列号二选一
        "deviceNo":"571****ALF"//设备序列号 设备Id序列号二选一
         "currPage":1,  //当前页 必选参数
         "pageSize":10 //每页返回的记录数，必选擦数，最多不能超过100
    }:
    '''
    deviceid = request.args.get('deviceid')
    json = "{\"userId\": \"200017377\", \"deviceId\": " + deviceid + ",\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getSingleDeviceDatas'
    s = getinfo(json, url)
    print(s)
    s = s.replace("null", "0")
    try:
        res = eval(s)
    except:
        print("error")
    return render_template("deviceinfo.html", device=res["device"], deviceid=deviceid)


'''报警历史'''


@app.route("/alarmrecord")
def getRecord():
    '''
    {
      "userId":200*****80,//用户Id 必选参数
      "deviceId":20****93,//设备Id 可选参数
      "sensorId":20****535,//传感器Id 可选参数
      "currPage":1,//当前页码 必选参数
      "pageSize":10//返回的数据条数，最大不能超过100条数据
    }
    :return:
    '''
    sensorid = request.args.get("id")
    json = "{\"userId\": \"200017377\", \"sensorId\": " + sensorid + ",\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/alarms/getAlarmsHistory'
    s = getinfo(json, url)
    s = s.replace("null", "0")
    try:
        res = eval(s)
        if res["dataList"]:
            text = res["dataList"]
            text2 = "\"" + str(text) + "\""
            return text2
    except:
        print("transfer error")
        return "0"


'''历史数据'''


@app.route("/history")
def getHistory():
    '''
    {
      "userId":20******80,//用户Id
      "deviceId":20******5,//设备Id
      "sensorId":20******62,//传感器Id
      "startDate" : "2019-10-27 00:00:00",//开始时间
      "endDate" : "2019-10-28 23:59:59",//结束时间
      "currPage":1,//当前页码
      "pageSize":10//返回的数据条数 最大不能超过100
    }
    return:
    '''
    flag = "00"
    '''请求传感器的历史数据'''
    t = datetime.datetime.now()
    t1 = t.strftime('%Y-%m-%d %H:%M:%S')
    startDate = "\"" + (t - datetime.timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S") + "\"";
    endDate = "\"" + t1 + "\""
    deviceid = request.args.get("deviceid")
    sensorid = request.args.get("sensorid")
    json = "{\"userId\": \"200017377\",\"deviceId\":" + deviceid + ", \"sensorId\": " + sensorid + ",\"startDate\":" + startDate + ",\"endDate\":" + endDate + ",\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getSensorHistroy'
    s = getinfo(json, url)
    s = s.replace("null", "0")
    try:
        res1 = eval(s)
        # print(res1)
    except:
        print("error")
    '''请求单个传感器的信息
        {
            "userId": 1******08,//用户Id必传参数
            "sensorId":20******44//传感器id 
        }
    '''
    json2 = "{\"userId\": \"200017377\", \"sensorId\": " + sensorid + " }"
    url2 = 'http://api.tlink.io/api/device/getSingleSensorDatas'
    s2 = getinfo(json2, url2)
    s2 = s2.replace("null", "0")
    try:
        res2 = eval(s2)
        flag = res2["flag"]
    except:
        flag = "01"
        print("error")
    return render_template("historySearch.html", res1=res1, res2=res2, startDate=startDate[1:-1], endDate=endDate[1:-1],
                           flag=flag)


'''历史查询
返回文本为具体数据，具体处理在前端的js
'''


@app.route("/historysearch")
def historysearch():
    deviceid = request.args.get("deviceid")
    sensorid = request.args.get("sensorid")
    startDate = "\"" + request.args.get("starttime") + "\""
    endDate = "\"" + request.args.get("endtime") + "\""
    json = "{\"userId\": \"200017377\",\"deviceId\":" + deviceid + ", \"sensorId\": " + sensorid + ",\"startDate\":" + startDate + ",\"endDate\":" + endDate + ",\"currPage\": 1,\"pageSize\": 100 }"
    url = 'http://api.tlink.io/api/device/getSensorHistroy'
    s = getinfo(json, url)
    s = s.replace("null", "0")
    text = "123"
    try:
        res = eval(s)
        if res["flag"] == "00":
            text = res["dataList"]
            text2 = "\"" + str(text) + "\""
            return text2
        else:
            return text
    except:
        print("transfer error")
    return text


'''质量评估模块概览'''


@app.route("/assess")
def assess():
    return render_template("assessFish.html")


'''质量评估木块实例'''


@app.route("/assessinstance")
def assessinstance():
    return render_template("assessinstance.html", flag=0, originpath="", result_path="", eye_fresh_level="",
                           eye_believeble=1,
                           gill_fresh_level="", gill_believeble=1, all_fresh_level="", eye_star_len="",
                           gill_star_len="", type=0)


@app.route("/assessyinchang")
def assessyinchang():
    return render_template("assessinstance.html", flag=0, originpath="", result_path="", eye_fresh_level="",
                           eye_believeble=1,
                           gill_fresh_level="", gill_believeble=1, all_fresh_level="", eye_star_len="",
                           gill_star_len="", type=1)


@app.route("/infocollect")
def infocollect():
    return render_template("infocollect.html")


@app.route("/infoinstance")
def infoinstance():
    try:
        connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                     charset="utf8")
        cursor = connection.cursor()
        # 在表中查询结果，用result保存
        sql = "select * from infodevice"
        cursor.execute(sql)
        result = cursor.fetchall()
        connection.close()
        length = len(result)
    except:
        print("error")
    return render_template("infoinstance.html", length=length, result=result)


@app.route("/infofolder")
def infofolder():
    devicename = request.args.get("devicename")
    try:
        connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                     charset="utf8")
        cursor = connection.cursor()
        # 在表中查询结果，用result保存
        sql = "select * from devicefolder where devicename='%s'" % (devicename)
        cursor.execute(sql)
        result = cursor.fetchall()
        connection.close()
        length = len(result)
    except:
        print("error")
    return render_template("infodevicefolder.html", length=length, result=result, devicename=devicename)


@app.route("/infofile")
def infofile():
    devicename = request.args.get("devicename")
    foldername = request.args.get("foldername")

    result, result1, result2 = getfilelist(devicename, foldername)
    length = len(result)
    length1 = len(result1)
    length2 = len(result2)
    return render_template("infodevicefile.html", length=length, result=result, length1=length1, result1=result1,
                           length2=length2, result2=result2, devicename=devicename, foldername=foldername, flag=0)


@app.route("/infofileupload", methods=['POST', 'GET'])
def infofileupload():
    devicename = request.args.get("devicename")
    foldername = request.args.get("foldername")
    '''查看上传的文件是否与当前目录下有重名'''
    flag = 0
    try:
        if request.method == 'POST':
            files = request.files.getlist('file')
            if files:
                piclist, donefiles, waitfiles = getfilelist(devicename, foldername)
                if len(piclist) != 0:
                    for file in files:
                        filename = secure_filename(file.filename)
                        for pic in piclist:
                            if filename == pic[2]:
                                flag = 1
                                break;
                        if flag == 1:
                            break;
                '''无重名，开始上传'''
                if flag == 0:
                    try:
                        connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root",
                                                     password="root",
                                                     charset="utf8")
                        cursor = connection.cursor()
                        basepath = os.path.dirname(__file__)  # 当前文件所在路径
                        for file in files:
                            filename = secure_filename(file.filename)
                            upload_path = os.path.join(basepath, 'static', 'uploadFileDir', devicename, foldername,
                                                       filename)
                            upload_path = os.path.abspath(upload_path)
                            absupload_path = os.path.join("../static/uploadFileDir/", devicename, foldername, filename)
                            file.save(upload_path)
                            print("upload", basepath, upload_path, "OK")
                            t = datetime.datetime.now()
                            time = t.strftime('%Y-%m-%d %H:%M:%S')
                            sql = "insert into devicefile(devicename,foldername,filename,filepath,isassess,uploadtime) values (%s,%s,%s,%s,%s,%s) "
                            # 执行SQL语句，返回数据库改变的记录数，正常情况下返回1
                            row = cursor.execute(sql, [devicename, foldername, filename, absupload_path, "0", time])
                            connection.commit()
                        connection.close()
                    except:
                        flag = 1
                        print("connect database error")
            else:
                flag = 2
    except:
        print("method error")
    '''查询重定向'''
    try:
        result, result1, result2 = getfilelist(devicename, foldername)
        length = len(result)
        length1 = len(result1)
        length2 = len(result2)
    except:
        print("error")
    return render_template("infodevicefile.html", length=length, result=result, length1=length1, result1=result1,
                           length2=length2, result2=result2, devicename=devicename, foldername=foldername, flag=flag)


@app.route("/delpic")
def delpic():
    flag = "1"
    devicename = request.args.get("devicename")
    foldername = request.args.get("foldername")
    delpiclist = request.args.get("picname")
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    if delpiclist:
        delpiclist = eval(delpiclist)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        for delpic in delpiclist:
            try:
                sql = "delete from devicefile where devicename='%s' and foldername = '%s' and filename = '%s'" % (
                devicename, foldername, delpic)
                cursor.execute(sql)
                connection.commit()
            except:
                print("删除数据库相应file出错")
            '''删除根目录下的文件'''
            picpath = os.path.join(project_path, devicename, foldername, delpic)
            respicpath = os.path.join(result_path, devicename, foldername, delpic)
            try:
                print("del" + picpath)
                os.remove(picpath)
                os.remove(respicpath)
            except:
                flag = "1"
                print("result pic error")
        connection.close()
    return flag


@app.route("/deldevice")
def deldevice():
    '''注意未将关联表数据删除'''
    deldevicelist = request.args.get("devicename")
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    flag = "1"
    if deldevicelist:
        deldevicelist = eval(deldevicelist)
        try:
            for deldevice in deldevicelist:
                sql = "delete from infodevice where devicename='%s'" % (deldevice)
                removedir(project_path + deldevice)
                removedir(result_path + deldevice)
                cursor.execute(sql)
                connection.commit()
            connection.close()
        except:
            flag = "0";
            print("del error")
    return flag


@app.route("/deldevicefolder")
def deldevicefolder():
    '''注意未将关联表数据删除'''
    devicename = request.args.get("devicename")
    delfolderlist = request.args.get("foldername")
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    flag = "1"
    if delfolderlist:
        delfolderlist = eval(delfolderlist)
        try:
            for delfolder in delfolderlist:
                '''删除相应设备下的文件夹'''
                removedir(os.path.join(project_path, devicename, delfolder))
                removedir(os.path.join(result_path, devicename, delfolder))
                sql = "delete from devicefolder where devicename='%s' and foldername='%s'" % (devicename, delfolder)
                cursor.execute(sql)
                connection.commit()
            connection.close()
        except:
            flag = "0";
            print("del error")
    return flag


@app.route("/adddevice")
def adddevice():
    flag = "1"
    newdevicename = request.args.get("devicename")
    print(newdevicename)
    t = datetime.datetime.now()
    time = t.strftime('%Y-%m-%d %H:%M:%S')
    owner = "yx"
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    try:
        path = project_path + newdevicename
        print("创建文件夹", path)
        os.mkdir(path)
        os.mkdir(result_path + newdevicename)
        sql = "insert into infodevice(devicename,createtime,owner) values (%s,%s,%s) "
        # 执行SQL语句，返回数据库改变的记录数，正常情况下返回1
        row = cursor.execute(sql, [newdevicename, time, owner])
        connection.commit()
        connection.close()
    except:
        flag = "0"
        print("create error")
    return flag


@app.route("/adddevicefolder")
def adddevicefolder():
    flag = "1"
    devicename = request.args.get("devicename")
    newfoldername = request.args.get("name")
    '''创建设备下的文件夹'''
    os.mkdir(os.path.join(project_path, devicename, newfoldername))
    os.mkdir(os.path.join(result_path, devicename, newfoldername))
    fishtype = request.args.get("type")
    t = datetime.datetime.now()
    time = t.strftime('%Y-%m-%d %H:%M:%S')
    owner = "yx"
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    try:
        sql = "insert into devicefolder(devicename,foldername,createtime,owner,fishtype) values (%s,%s,%s,%s,%s) "
        # 执行SQL语句，返回数据库改变的记录数，正常情况下返回1
        row = cursor.execute(sql, [devicename, newfoldername, time, owner, fishtype])
        connection.commit()
        connection.close()
    except:
        flag = "0"
        print("create error")
    return flag


@app.route("/wait")
def wait():
    return render_template("wait.html")
