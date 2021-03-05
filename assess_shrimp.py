import os

from flask import request, jsonify, Blueprint, render_template
from werkzeug.utils import secure_filename

from ML import ML

# 创建蓝图
assess_shrimp_page = Blueprint("app_assess_shrimp", __name__)

# 全局配置
UPLOAD_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXT = ('xlsx')
path = UPLOAD_DIR + '/static/result/other'
print(UPLOAD_DIR)

print(path)


# 判断文件后缀名
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT
    #  rsplit() 方法通过指定分隔符对字符串进行分割并返回一个列表


# 文件上传并评估，返回结果
@assess_shrimp_page.route('/access_shrimp/', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传文件，您只能上传.xlsx格式的文件"})
        upload_path = os.path.join(path, secure_filename(f.filename))
        f.save(upload_path)
        print("上传好了")
        print(upload_path)
        # 评估结果展示
        files = os.listdir(path)
        fileList = {}
        # 执行模型脚本
        os.popen("python ./ML/ML.py -i %s" % upload_path)
        result = ML.process(upload_path)
        # 将字典形式的数据转化为字符串
        print(result)
        workpath = os.getcwd()
        dir = workpath + "/static/result/result.png"
    return render_template("assessinstance.html", result=result, dir=dir, flag=1)
