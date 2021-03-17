from flask import Flask,render_template,Blueprint,request
import sys
import os
import skimage.io
from getDataAPI.filemanage import getfilelist,getfishtype
from werkzeug.utils import secure_filename
from Mask_RCNN.mrcnn.config import Config
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import visualize
import imutils
import numpy as np
import pymysql
import pdfkit
app = Blueprint("app_assess", __name__)
project_path= "C:/Work/Fish/static/uploadFileDir/"
res_path = "C:/Work/Fish/static/result/"
pdf_path = "C:/Work/Fish/templates/"
'''上传评估图片文件'''
class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 8
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 1024
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)
    TRAIN_ROIS_PER_IMAGE = 100
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 10

class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
ROOT_DIR = os.path.abspath("../Fish//Mask_RCNN")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples\\coco\\"))
MODEL_DIR = os.path.join(ROOT_DIR, "logs\\yinchangmodel")
MOEDL_JIN_DIR = os.path.join(ROOT_DIR,"logs\\jinchangmodel")
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0040.h5")
COCO_JIN_MOEDL_PATH = os.path.join(MOEDL_JIN_DIR,"mask_rcnn_shapes_0048.h5")
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model1 = modellib.MaskRCNN(mode="inference", model_dir=MOEDL_JIN_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)
model1.load_weights(COCO_JIN_MOEDL_PATH,by_name=True)
model.keras_model._make_predict_function()
model1.keras_model._make_predict_function()

'''评测页面响应上传评估事件'''
@app.route("/upload",methods=['POST','GET'])
def upload():
    fishtype = request.args.get("type")
    print(fishtype)
    if request.method=='POST':
        '''未上传图片'''
        try:
            f = request.files['file']
        except:
            return render_template("assessinstance.html",flag=0,originpath  = "",result_path = "",eye_fresh_level="",eye_believeble=1,
                               gill_fresh_level="",gill_believeble=1,all_fresh_level="",eye_star_len="",gill_star_len="",type=0)
        '''上传图片为之前定义的加号图片'''
        if f.filename == "plus.jpg":
            return render_template("assessinstance.html",flag=0,originpath  = "",result_path = "",eye_fresh_level="",eye_believeble=1,
                               gill_fresh_level="",gill_believeble=1,all_fresh_level="",eye_star_len="",gill_star_len="",type=0)
        else:
            '''将上传的文件保存到uploadFileDir'''
            basepath = os.path.dirname(__file__) #当前文件所在路径
            upload_path = os.path.join(basepath,'static','uploadFileDir/other',secure_filename(f.filename))
            upload_path = os.path.abspath(upload_path)
            print(basepath,upload_path)
            f.save(upload_path)
            '''开始预测部分，加载模型'''
            result_path = os.path.join(basepath,'static',"result/other",f.filename)
            class_names = ['BG', 'gill1', 'eye1', 'gill2', 'eye2', 'gill3', 'eye3', 'gill4', 'eye4']
            fresh_level = ['新鲜','较为新鲜','不新鲜','腐败']
            image = skimage.io.imread(upload_path)
            image = imutils.resize(image,width=1024,height=640,inter=3)#对image进行resize
            # Run detection
            if fishtype=="1":#model为银鲳 fishtype=1
                print("调用银鲳预测模型")
                results = model.detect([image], verbose=1)
            else: #model1为金鲳 fishtype=0
                print("调用金鲳预测模型")
                results = model1.detect([image],verbose=1)
            # Visualize results
            r = results[0]
            #print(r)

            class_ids = r['class_ids']
            scores = r['scores']
            print(class_ids,scores)
            eye_believeble = 0
            gill_believeble = 0
            eye_fresh_level = "新鲜"
            gill_fresh_level = "新鲜"
            eye_star_len = np.zeros(4)
            gill_star_len = np.zeros(4)
            for i in range(0,len(class_ids)):
                j = class_ids[i]
                if j%2==0:
                    eye_fresh_level = fresh_level[int(j/2)-1]
                    eye_star_len = 4 - (int(j/2)-1)
                    eye_believeble = scores[i]
                    eye_star_len = np.zeros(eye_star_len)
                elif j%2==1:
                    gill_fresh_level = fresh_level[int(j/2)]
                    gill_star_len = 4 - int(j/2)
                    gill_star_len = np.zeros(gill_star_len)
                    gill_believeble = scores[i]
            visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'],result_path,f.filename)
            if eye_believeble>gill_believeble:
                all_fresh_level = eye_fresh_level
            else:
                all_fresh_level = gill_fresh_level
            '''将路径下的图片传到前端'''
            origin_path = os.path.join('..\\static','uploadFileDir\\other',f.filename)
            res_path = os.path.join('..\\static','result\\other',f.filename)
        return render_template("assessinstance.html",flag=1,originpath  = origin_path,result_path = res_path,
                               eye_fresh_level=eye_fresh_level,eye_believeble=eye_believeble,
                               gill_fresh_level=gill_fresh_level,gill_believeble=gill_believeble,
                               all_fresh_level = all_fresh_level,eye_star_len=eye_star_len,
                               gill_star_len=gill_star_len,type=0)
    else:
        return render_template("assessinstance.html",flag=0,originpath  = "",result_path = "",eye_fresh_level="",eye_believeble=1,
                               gill_fresh_level="",gill_believeble=1,all_fresh_level=""
                               ,eye_star_len="",gill_star_len="",type=0)
@app.route("/assessallfish")
def assessallfish():
    flag="1"
    devicename = request.args.get("devicename")
    foldername = request.args.get("foldername")
    fishtype = getfishtype(devicename,foldername)
    print(fishtype)
    result,result1,result2 = getfilelist(devicename,foldername)
    class_names = ['BG', 'gill1', 'eye1', 'gill2', 'eye2', 'gill3', 'eye3', 'gill4', 'eye4']
    fresh_level = ['新鲜', '较为新鲜', '不新鲜', '腐败']
    connection = pymysql.connect(host="127.0.0.1", port=3306, db="fish", user="root", password="root",
                                 charset="utf8")
    cursor = connection.cursor()
    for res in result2:
        '''开始预测部分，加载模型'''
        file_path = os.path.join(project_path,devicename,foldername,res[2])
        result_path = os.path.join(res_path, devicename, foldername, res[2])
        image = skimage.io.imread(file_path)
        image = imutils.resize(image, width=1024, height=640, inter=3)  # 对image进行resize
        # Run detection
        if fishtype == 1:  # model为银鲳 fishtype=1
            print("调用银鲳预测模型")
            results = model.detect([image], verbose=1)
        else:  # model1为金鲳 fishtype=0
            print("调用金鲳预测模型")
            results = model1.detect([image], verbose=1)
        # Visualize results
        r = results[0]
        # print(r)
        class_ids = r['class_ids']
        scores = r['scores']
        print(class_ids, scores)
        eye_believeble = 0
        gill_believeble = 0
        eye_fresh_level = "新鲜"
        gill_fresh_level = "新鲜"
        eye_star_len = 4
        gill_star_len = 4
        for i in range(0, len(class_ids)):
            j = class_ids[i]
            if j % 2 == 0:
                eye_fresh_level = fresh_level[int(j / 2) - 1]
                eye_star_len = 4 - (int(j / 2) - 1)
                eye_believeble = scores[i]
            elif j % 2 == 1:
                gill_fresh_level = fresh_level[int(j / 2)]
                gill_star_len = 4 - int(j / 2)
                gill_believeble = scores[i]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'], result_path, res[2])
        if eye_believeble > gill_believeble:
            all_fresh_level = eye_fresh_level
            if eye_star_len!=gill_star_len:
                gill_star_len=eye_star_len
        else:
            all_fresh_level = gill_fresh_level
            if eye_star_len!=gill_star_len:
                eye_star_len=gill_star_len
        print(all_fresh_level)
        '''更新数据库'''
        try:
            sql = "update devicefile set isassess = '%d',eyefresh='%s',gillfresh='%s',allfresh='%s',eyebelievable='%lf',gillbelievable='%lf',gillstar='%d',eyestar='%d' " \
                  "where devicename='%s' and foldername='%s' and filename='%s'"%(1,eye_fresh_level,gill_fresh_level,all_fresh_level,eye_believeble,gill_believeble,gill_star_len,eye_star_len,devicename,foldername,res[2])
            print(sql)
            cursor.execute(sql)
            connection.commit()
        except:
            flag="0"
            print("更新数据库失败")
    connection.close()
    return flag

@app.route("/showassessresult")
def assessresult():
    devicename = request.args.get("devicename")
    foldername = request.args.get("foldername")
    result,result1,result2 = getfilelist(devicename,foldername)
    length =len(result1)
    return render_template("assessreport.html",result=result1,length=length)


@app.route("/savepdf")
def generatepdf():
    flag="0"
    filename="wait.html"
    pdfkit.from_file(os.path.join(pdf_path,filename), 'out.pdf')
    return flag