import pymysql
from flask import Blueprint, render_template, request

from getDataAPI.api import generate_token, certify_token

app = Blueprint("app_usercontrol", __name__)


@app.route("/login")
def login():
    msg = ""
    return render_template("login.html", msg=msg)


@app.route("/logincheck", methods=['POST', 'GET'])
def logincheck():
    msg = ""
    try:
        if request.method == 'POST':
            username = request.form.get("username")
            passwd = request.form.get("pwd")
            if username == "" or passwd == "":
                msg = "用户名或密码为空"
            else:
                try:
                    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                                 charset="utf8")
                    cursor = connection.cursor()
                    sql = "select * from login where username='%s' and password='%s'" % (username, passwd)
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    connection.close()
                    length = len(result)
                    if length == 0:
                        msg = "用户名或密码错误"
                    else:
                        key = username + passwd
                        token = generate_token(key, 3600)
                        msg = "登录成功"
                        print(key, token)
                        return render_template("index.html", token=token, key=key, operator="login")
                except:
                    print("connect database error")
    except:
        print("methods error")
    return render_template("login.html", msg=msg)


@app.route("/quit")
def quit():
    token = ""
    key = ""
    return render_template("index.html", token=token, key=key, operator="quit")


@app.route("/confirmtoken")
def confirmtoken():
    key = request.args.get("key")
    token = request.args.get("token")
    flag = certify_token(key, token)
    if flag:
        flag = "true"
    else:
        flag = "false"
    return flag


@app.route("/register")
def register():
    return render_template("register.html")
