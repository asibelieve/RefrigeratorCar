# Python简单删除目录下文件以及文件夹
import os
import shutil

import pymysql


def removedir(rootdir):
    filelist = os.listdir(rootdir)  # 列出该目录下的所有文件夹
    for f in filelist:
        filepath = os.path.join(rootdir, f)  # 将文件名映射成绝对路劲
        if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
            os.remove(filepath)  # 若为文件，则直接删除
            print(str(filepath) + " removed!")
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)  # 若为文件夹，则删除该文件夹及文件夹内所有文件
            print("dir " + str(filepath) + " removed!")
    shutil.rmtree(rootdir, True)  # 最后删除img总文件夹
    print("删除成功")


def getfilelist(devicename, foldername):
    result = []
    try:
        connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                     charset="utf8")
        cursor = connection.cursor()
        # 在表中查询结果，用result保存
        sql = "select * from devicefile where devicename='%s' and foldername = '%s'" % (devicename, foldername)
        cursor.execute(sql)
        result = cursor.fetchall()
        sql1 = "select * from devicefile where devicename='%s' and foldername = '%s' and isassess=1" % (
        devicename, foldername)
        cursor.execute(sql1)
        result1 = cursor.fetchall()
        sql2 = "select * from devicefile where devicename='%s' and foldername = '%s' and isassess=0" % (
        devicename, foldername)
        cursor.execute(sql2)
        result2 = cursor.fetchall()
        connection.close()
    except:
        print("error")
    return result, result1, result2;


def getfishtype(devicename, foldername):
    try:
        connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                     charset="utf8")
        cursor = connection.cursor()
        # 在表中查询结果，用result保存
        sql = "select * from devicefolder where devicename='%s' and foldername = '%s'" % (devicename, foldername)
        cursor.execute(sql)
        result = cursor.fetchone()
        fishtype = result[4]
        connection.close()
    except:
        print("error")
    return fishtype
