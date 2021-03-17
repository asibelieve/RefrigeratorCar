from flask import Flask

from app import app as app1
from assess_meat import assess_meat_page
from assess_shrimp import assess_shrimp_page
from manage import app as appmanage
from usercontrol import app as appusercontrol
from assess import app as appassess

# 创建flask对象
app = Flask(__name__)
# 使用blueprint注册之前创建的flask对象
app.register_blueprint(app1)
app.register_blueprint(appmanage)
app.register_blueprint(appassess)
app.register_blueprint(appusercontrol)
app.register_blueprint(assess_shrimp_page)
app.register_blueprint(assess_meat_page)
if __name__ == '__main__':
    app.run()
