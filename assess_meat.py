from flask import request, Blueprint, render_template

from Meat import meat

# 创建蓝图
assess_meat_page = Blueprint("app_assess_meat", __name__)


# 用户填写三个参数，然后去执行肉质新鲜度预测程序，得到返回的预测结果。
@assess_meat_page.route('/assess_meat_submit/', methods=['POST'], strict_slashes=False)
def access_meat():
    tem = request.form.get("tem")
    o2 = request.form.get("o2")
    nh3 = request.form.get("nh3")

    # 执行模型脚本
    result = meat.fresh_level(tem, o2, nh3)

    return render_template("assess_meat.html", result=result, flag=1)
