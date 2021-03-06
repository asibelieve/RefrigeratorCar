# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 21:20:46 2021

@author: Administrator
"""
import os
import random

import matplotlib.pyplot as plt
import pandas as pd

global data_ysbc, data_ystvbn, data_ysgp, data_test, data_ys
from sklearn.decomposition import PCA

global data_ysgp_msc  # 多元散射校正MSC
global data_ysgp_snv  # 标准正态变量交化SNV
global data_ysgp_Nor  # % 归一化Normalize
global data_ysgp_MCX  # % 数据中心化(Mean centering)
global data_ysgp_ax  # 标准化Autoscales
global data_ysgp_nir  # (11点)移动窗口平滑/Moving-average method
global data_ysgp_S_G  # SavitZky一Golay卷积平滑法及求导
global data_ysgp_S_G1  # SavitZky-Golay一阶求导
global data_ysgp_S_G2  # SavitZky-Golay二阶求导
import numpy as np
import matplotlib

for f in matplotlib.font_manager.fontManager.ttflist:
    basePath = os.getcwd() + '/ML/'


def data_divide(data):  # 定义数据分割函数，将数据分为训练集和测试集
    r1 = data.shape[0]
    n = list(range(0, r1))
    random.shuffle(n)
    n_choose = round(r1 * 0.75)
    data_ys_train = data.iloc[n[0:n_choose], :]
    data_ys_test = data.iloc[n[n_choose:r1], :]
    return data_ys_train, data_ys_test, n, n_choose


def tongji(data_ys_train, data_ys_test):
    ys_train_nsample = data_ys_train.shape[0]  # 校正集的样本数
    ys_train_mean = np.mean(data_ys_train.iloc[:, -1])  # 校正集的平均值
    ys_train_max = max(data_ys_train.iloc[:, -1])  # 校正集的最大值
    ys_train_min = min(data_ys_train.iloc[:, -1])  # 校正集的最小值
    ys_train_std = np.std(data_ys_train.iloc[:, -1])  # 校正集的标准偏差

    ys_test_nsample = data_ys_test.shape[0]  # 测试集的样本数
    ys_test_mean = np.mean(data_ys_test.iloc[:, -1])  # 测试集的平均值
    ys_test_max = max(data_ys_test.iloc[:, -1])  # 测试集的最大值
    ys_test_min = min(data_ys_test.iloc[:, -1])  # 测试集的最小值
    ys_test_std = np.std(data_ys_test.iloc[:, -1])  # 测试集的标准偏差

    return [ys_train_nsample, ys_train_mean, ys_train_max, ys_train_min, ys_train_std, ys_test_nsample, ys_test_mean,
            ys_test_max, ys_test_min, ys_test_std]


def msc(x, xref):  # 多元散射校正MSC
    m, n = x.shape
    rs = np.array(xref).reshape(n, 1)
    cw = np.ones((1, n))
    mz = np.concatenate([cw.T, rs], axis=1)
    mm, nm = mz.shape
    z = np.dot(mz.T, mz)
    u, s, v = np.linalg.svd(z)
    cn = pow(10, 12)
    ms = s[0] / np.sqrt(cn)
    #    cs=max(s,ms)
    cz = np.dot(u, np.diag(s))
    cz = np.dot(cz, v.T)
    zi = np.linalg.inv(cz)

    b = np.dot(zi, mz.T)
    b = np.dot(b, x.T)
    B = b.T
    x_msc1 = x
    p = B[:, 0]
    low = np.dot(p.reshape(m, 1), np.ones((1, mm)))
    x_msc1 = x_msc1 - low
    p = B[:, 1]
    low = np.dot(p.reshape(m, 1), np.ones((1, mm)))
    x_msc = x_msc1 / low
    return x_msc


def snv(x):
    m, n = x.shape
    rmean = np.mean(x, axis=1)
    rmean = np.array(rmean).reshape(m, 1)
    dr = x - np.repeat(rmean, n, axis=1)
    ss = np.sqrt(np.sum(pow(dr, 2), axis=1) / (n - 1))
    x_snv = dr / np.repeat(np.array(ss).reshape(m, 1), n, axis=1)
    return x_snv


def normaliz(x):
    m, n = x.shape
    nx = x.copy()
    nm = np.linalg.norm(np.array(nx), axis=1)
    for i in range(m):
        nx.iloc[i, :] = nx.iloc[i, :] / nm[i]
    return nx


def center(x):
    m, n = x.shape
    mx = np.mean(x, axis=0)
    mcx = x.copy()
    for i in range(m):
        for j in range(n):
            mcx.iloc[i][j] = mcx.iloc[i][j] - mx[j]
    return mcx, mx


def auto(x):
    m, n = x.shape
    mx = np.mean(x, axis=0)
    stdx = np.std(x, axis=0)
    ax = x.copy()
    for i in range(m):
        for j in range(n):
            ax.iloc[i][j] = (ax.iloc[i][j] - mx[j]) / stdx[j]
    return ax, mx, stdx


def nirmaf(data, window):
    m, n = data.shape
    mdata = np.zeros((m, n))
    wcenter = np.floor(window / 2)
    wcenter = int(wcenter)
    extdata = np.concatenate([np.zeros((m, wcenter)), data, np.zeros((m, wcenter))], axis=1)
    # 延展后的光谱矩阵曲线拟合
    extdata = pd.DataFrame(extdata)
    for k in range(m):
        bstart = np.polyfit(np.arange(wcenter + 1, wcenter + 5), extdata.iloc[k, wcenter:wcenter + 4], 2)  # 左端曲线拟合
        bend = np.polyfit(np.arange(n - 4 + wcenter, n + wcenter + 1), extdata.iloc[k, n - 5 + wcenter:n + wcenter],
                          2)  # 左端曲线拟合
        extdata.iloc[k, 0:wcenter] = np.polyval(bstart, np.arange(1, wcenter + 1))  # 返回左端曲线拟合的值
        extdata.iloc[k, n + wcenter:n + window - 1] = np.polyval(bend,
                                                                 np.arange(n + wcenter + 1, n + window))  # 返回右端曲线拟合的值
    # 均值平滑
    for i in range(n):
        mdata[:, i] = np.mean(extdata.iloc[:, i:i + window].T).T
    return mdata


# def savgol_1(x,width):
#    m,n=x.shape
#    x_sg=x.copy()
#    order=2 
#    deriv=0
#    w=max(3,(1+2*round((width-1)/2)))
#    o=min([max(0,round(order)),5,w-1])
#    d=min(max(0,round(deriv)),o)
#    p=int(np.floor((w-1)/2))
#    xc1=np.dot(np.array(range(-p,p+1)).reshape(2*p+1,1),np.ones((1,1+o)))
#    xc2=np.ones((1,w)).reshape(w,1)*np.array(range(o+1)).reshape(1,o+1)
#    xc=pow(xc1,xc2)
#    c=np.eye(w)
#    we=np.dot(np.linalg.inv(c),xc)
#    we=np.linalg.lstsq(xc,c,rcond=None)[0]
#    b1=np.dot(np.ones((d,1)).reshape(d,1),np.array(range(1,o+1-d+1)).reshape(1,o+1-d))
#    b2=np.dot((np.array(range(d)).reshape(d,1)),np.ones((1,o+1-d)).reshape(1,o+1-d))
#    b=np.prod(b1+b2,axis=0)
#    di1=np.dot(np.ones((n,1)),np.array(we[d,:]).reshape(1,we.shape[1]))*b[0]
#    di=spdiags(di1,np.array(range(-114,114))[::-1],n,n).toarray()
#    di=spdiags(ones(n,1)*we(d+1,:)*b(1),p:-1:-p,n,n).toarray()
#    w1=diag(b)*we(d+1:o+1,:)
#    di(1:w,1:p+1)=[xc(1:p+1,1:1+o-d)*w1]' 
#    di(n-w+1:n,n-p:n)=[xc(p+1:w,1:1+o-d)*w1]'
#    x_sg=x*di
#    return x_sg
#
# def savgol(x,width,order,deriv):
#    m,n=x.shape
#    x_sg=x
#    w=max(3,(1+2*round((width-1)/2)))
#    o=min([max(0,round(order)),5,w-1])
#    d=min(max(0,round(deriv)),o)
#    p=(w-1)/2
#    xc=
#    xc=((-p:p)'*ones(1,1+o)).^(ones(size(1:w))'*(0:o))
#    we=xc\eye(w)
#    b=prod(ones(d,1)*[1:o+1-d]+[0:d-1]'*ones(1,o+1-d,1),1)
#    di=spdiags(ones(n,1)*we(d+1,:)*b(1),p:-1:-p,n,n)
#    w1=diag(b)*we(d+1:o+1,:)
#    di(1:w,1:p+1)=[xc(1:p+1,1:1+o-d)*w1]' 
#    di(n-w+1:n,n-p:n)=[xc(p+1:w,1:1+o-d)*w1]'
#    x_sg=x*di
#    return x_sg


def norm_pca(x):
    data_ysgp_stdr = np.std(x, axis=0)
    data_ysgp_row = x.shape[0]
    norm_data_ysgp = x / np.repeat(np.array(data_ysgp_stdr).reshape(1, x.shape[1]), data_ysgp_row, axis=0)
    return norm_data_ysgp


def data_pretreatment(data_ysbc, data_ysgp):
    '''data_ysgp表示n×p维定标反射率矩阵，n为样品数，p为波点数'''

    data_ysgp_mean = np.mean(data_ysgp)  # 表示所有样品的光谱在各个波长点处求平均值
    data_ysgp_row, data_ysgp_col = data_ysgp.shape

    # 多元散射校正MSC
    global data_ysgp_msc
    data_ysgp_msc = msc(data_ysgp, data_ysgp_mean)
    #    plt.figure(2)
    #    for iNsample in range(data_ysgp_row):
    #        plt.plot(x_lim,data_ysgp_msc.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    plt.title('多元散射校正(MSC)')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.show()

    # 标准正态变量交换SNV
    global data_ysgp_snv
    data_ysgp_snv = snv(data_ysgp)

    #    plt.figure(3)
    #    for iNsample in range(data_ysgp_row):
    #        plt.plot(x_lim,data_ysgp_snv.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    plt.title('标准正态变量交换(SNV)')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()

    # 归一化Normalize
    global data_ysgp_Nor
    data_ysgp_Nor = normaliz(data_ysgp)
    #    plt.figure(4)
    #    for iNsample in range(data_ysgp_row):
    #      plt.plot(x_lim,data_ysgp_Nor.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    plt.title('归一化Normalize')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()

    # 数据中心化(Mean centering)
    global data_ysgp_MCX, data_ysgp_MX
    data_ysgp_MCX, data_ysgp_MX = center(data_ysgp)

    #    plt.figure(5)
    #    for iNsample in range(data_ysgp_row):
    #        plt.plot(x_lim,data_ysgp_MCX.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    plt.title('数据中心化(Mean centering)')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #
    #    plt.show()

    # 标准化Autoscales
    global data_ysgp_ax
    data_ysgp_ax, mx, stdx = auto(data_ysgp.T)
    data_ysgp_ax = pd.DataFrame(data_ysgp_ax.T)
    #    plt.figure(6)
    #    for iNsample in range(data_ysgp_row):
    #        plt.plot(x_lim,data_ysgp_ax.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    plt.title('标准化Autoscales')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    ##    plt.show()
    #    plt.show()

    '''移动窗口平滑/Moving-average method'''
    global data_ysgp_nir
    data_ysgp_nir = pd.DataFrame(nirmaf(data_ysgp, 11))

    #    plt.figure(7)
    #
    #    for iNsample in range(data_ysgp_row):
    #        plt.plot(x_lim,data_ysgp_nir.iloc[iNsample,:],'-r',linewidth=0.5)
    #
    #    #xlim([data_ysbc(1) data_ysbc(end)])
    #
    #    plt.title('移动窗口平滑光谱')
    #    plt.xlabel('波长/nm Wavelength')
    #    plt.ylabel('反射率')
    #    plt.show()

    '''卷积平滑法及求导'''
    #    data_ysgp_S_G =savgol(data_ysgp,15)#光谱，窗口大小；多项式项数；一阶求导;平滑;
    #    data_ysgp_S_G1 =savgol(data_ysgp,7,3,1)# 一阶求导
    #    data_ysgp_S_G2 =savgol(data_ysgp,7,3,2)#二阶求导
    global data_ysgp_S_G, data_ysgp_S_G1, data_ysgp_S_G2
    data_ysgp_S_G = pd.DataFrame(pd.read_excel(io=(os.path.join(basePath, 'data_ysgp_S_G.xlsx')), header=None))
    data_ysgp_S_G1 = pd.DataFrame(pd.read_excel(io=(os.path.join(basePath, 'data_ysgp_S_G1.xlsx')), header=None))
    data_ysgp_S_G2 = pd.DataFrame(pd.read_excel(io=(os.path.join(basePath, 'data_ysgp_S_G2.xlsx')), header=None))


#    plt.figure(8)
#    for iNsample in range(data_ysgp_row):
#        plt.plot(x_lim,data_ysgp_S_G.iloc[iNsample,:],'-r',linewidth=0.5)
#    plt.title('Savitzky-Golay卷积平滑法')
#    plt.xlabel('波长/nm Wavelength')
#    plt.ylabel('反射率')
#    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
#    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#    plt.show()


# SavitZky-Golay一阶求导
#    plt.figure(9)
#    
#    for iNsample in range(data_ysgp_row):
#        plt.plot(x_lim,data_ysgp_S_G1.iloc[iNsample,:],'-r',linewidth=0.5)        
#    plt.title('Savitzky-Golay一阶求导')
#    plt.xlabel('波长/nm Wavelength')
#    plt.ylabel('反射率')
#    plt.show()


# Savitzky-Golay二阶求导
#    plt.figure(10)
#    
#    for iNsample in range(data_ysgp_row):
#        plt.plot(x_lim,data_ysgp_S_G2.iloc[iNsample,:],'-r',linewidth=0.5)
#        
#    plt.title('Savitzky-Golay二阶求导')
#    plt.xlabel('波长/nm Wavelength')
#    plt.ylabel('反射率')
#    plt.show()


'''MCX_erase_PLS——剔除异常点和PLS模型预测
explain：数据中心化MCX的最佳异常点剔除,不断调整权重，
获取最佳的PLS的RMSEC_MCX情况下的最佳weight，PLS的建模参数，剔除掉异常点后的数据'''


def Distance_maha(data, weight, lmd):  # 马氏距离函数
    data_mean = np.mean(data, axis=0)
    data_row = data.shape[0]
    Dis_out = np.zeros((data_row, 1))  # % 开一个行数为样本数的矩阵Dis_out存放马氏距离
    Dis1 = pow((data - np.repeat(np.array(data_mean).reshape(1, data.shape[1]), data_row, axis=0)), 2)
    Dis2 = Dis1 / np.repeat(lmd, data_row, axis=0)  #
    Dis_out = np.sqrt(np.sum(Dis2, axis=1))  #
    # %% 阈值设定
    Dis_mean = np.mean(Dis_out)  # 马氏距离的均值
    Dis_std = np.std(Dis_out)  # 马氏距离的标准差
    Threshold = Dis_mean + weight * Dis_std  # 设置阀值
    # 剔除

    correctindex = []
    eraseindex = []
    for indexx in range(data_row):
        if Dis_out[indexx] <= Threshold:
            correctindex.append(indexx)
        else:
            eraseindex.append(indexx)
    erase_N = len(eraseindex)
    return Dis_out, correctindex, eraseindex, erase_N, Threshold


def MSC_erase_PLS(test):
    #    plt.figure(12)
    #    ymax =38.53
    #    ymsc_trhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='A',header = None)
    #    Y0=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='B',header = None)
    #
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),ymsc_trhat.iloc[:,0],Y0.iloc[:,0],'or')
    #
    #    plt.title('MSC和PLSR的训练集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #    plt.figure(13)
    #    ymsc_testhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='C',header = None)
    #    Y0_test=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='D',header = None)
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),ymsc_testhat.iloc[:,0],Y0_test.iloc[:,0],'*r')
    #    plt.title('MSC和PLSR的验证集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #    plt.figure(14)
    #    xrow_msc=107
    #    best_threshold_msc=2.7915
    #    Dis_msc=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='E',header = None)
    #    plt.plot(list(range(xrow_msc)),Dis_msc,'bo')
    #    plt.plot(np.arange(1,xrow_msc,0.1),np.repeat(np.array(best_threshold_msc).reshape(1,1),len(np.arange(1,xrow_msc,0.1)),axis=0),'--r',LineWidth=2)
    #    plt.title('多元散射校正MSC的马氏距离和最佳阈值')
    #    plt.xlabel('样品序号')
    #    plt.ylabel('马氏距离')
    #    plt.xlim([0,xrow_msc])  # x轴边界
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    ch0 = 325.1193
    ch0_mscte = np.repeat(np.array(ch0).reshape(1, 1), len(test), axis=0)
    xish = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AJ', header=None)
    ymsc_testhat = ch0_mscte + np.dot(test, xish)
    return ymsc_testhat


def Nor_erase_PLS(test):
    #    plt.figure(15)
    #    ymax =38.53
    #    yNor_trhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='F',header = None)
    #    Y0=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='G',header = None)
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),yNor_trhat.iloc[:,0],Y0.iloc[:,0],'om')
    #    plt.title('归一化和PLSR的训练集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel ('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #    plt.figure(16)
    #    yNor_testhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='H',header = None)
    #    Y0_test=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='I',header = None)
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),yNor_testhat.iloc[:,0],Y0_test.iloc[:,0],'*r')
    #    plt.title('归一化和PLSR的验证集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #
    #     #作图(马氏距离——最佳阈值图）
    #    plt.figure(17)
    #    xrow_Nor=107
    #    best_threshold_Nor=2.1087
    #    Dis_msc=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='J',header = None)
    #    plt.plot(list(range(xrow_Nor)),Dis_msc,'bo')
    #    plt.plot(np.arange(1,xrow_Nor,0.1),np.repeat(np.array(best_threshold_Nor).reshape(1,1),len(np.arange(1,xrow_Nor,0.1)),axis=0),'--r',LineWidth=2)
    #    plt.title('归一化的马氏距离和最佳阈值')
    #    plt.xlabel('样品序号')
    #    plt.ylabel('马氏距离')
    #    plt.xlim([0,xrow_Nor])  # x轴边界
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    ch0 = -55.8948
    ch0_norte = np.repeat(np.array(ch0).reshape(1, 1), len(test), axis=0)
    xish = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AK', header=None)
    ymsc_testhat = ch0_norte + np.dot(test, xish)
    return ymsc_testhat


def MCX_erase_PLS():  # 剔除异常点和PLS模型预测

    # 画出MCX训练集图谱
    plt.figure(18)
    ymax = 38.53
    yMCX_trhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='K', header=None)
    Y0 = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='L', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), yMCX_trhat.iloc[:, 0], Y0.iloc[:, 0], 'om')
    plt.title('数据中心化和PLSR的训练集预测结果')
    plt.xlabel('True Value（mgN/100g）')
    plt.ylabel('Predictive Value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    ymax = 34.87
    plt.figure(19)
    yMCX_testhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='M', header=None)
    Y0_test = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='N', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), yMCX_testhat.iloc[:, 0], Y0_test.iloc[:, 0], '*g')
    plt.title('数据中心化和PLSR的验证集预测结果')
    plt.xlabel('True Value（mgN/100g）')
    plt.ylabel('Predictive Value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    plt.figure(20)
    xrow_MCX = 107
    best_threshold_Msc = 3.5386
    Dis_mcx = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='O', header=None)
    plt.plot(list(range(xrow_MCX)), Dis_mcx, 'bo')
    plt.plot(np.arange(1, xrow_MCX, 0.1),
             np.repeat(np.array(best_threshold_Msc).reshape(1, 1), len(np.arange(1, xrow_MCX, 0.1)), axis=0), '--r',
             LineWidth=2)
    plt.title('数据中心化的马氏距离和最佳阈值')
    plt.xlabel('样品序号')
    plt.ylabel('马氏距离')
    plt.xlim([0, xrow_MCX])  # x轴边界
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()


def nir_erase_PLS():  # ——剔除异常点和PLS模型预测
    plt.figure(21)
    ymax = 38.53
    ynir_trhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='P', header=None)
    Y0 = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='Q', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), ynir_trhat.iloc[:, 0], Y0.iloc[:, 0], 'oc')
    plt.title('(11 points) moving window smoothing and PLSR training set prediction results')
    plt.xlabel('True Value（mgN/100g）')
    plt.ylabel('Predictive Value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    ymax = 34.87
    plt.figure(22)
    ynir_testhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='R', header=None)
    Y0_test = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='S', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), ynir_testhat.iloc[:, 0], Y0_test.iloc[:, 0], '*g')
    plt.title('(11点)移动窗口平滑和PLSR的验证集预测结果')
    plt.xlabel('True Value（mgN/100g）')
    plt.ylabel('Predictive Value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    # 
    # corr = corrcoef(ynir_testhat,Y0_test);
    # 
    # Rtest_nir =corr(1,2)
    # 
    # RMSEP_nir = sqrt((sum((ynir_testhat-Y0_test).^2))/num_nirtest)
    # 
    # 作图(马氏距离——最佳阈值图）
    plt.figure(23)
    xrow_nir = 107
    best_threshold_nir = 3.8372
    Dis_nir = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='T', header=None)
    plt.plot(list(range(xrow_nir)), Dis_nir, 'bo')
    plt.plot(np.arange(1, xrow_nir, 0.1),
             np.repeat(np.array(best_threshold_nir).reshape(1, 1), len(np.arange(1, xrow_nir, 0.1)), axis=0), '--r',
             LineWidth=2)
    plt.title('(11点)移动窗口平滑的马氏距离和最佳阈值')
    plt.xlabel('样品序号')
    plt.ylabel('马氏距离')
    plt.xlim([0, xrow_nir])  # x轴边界
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()


def snv_erase_PLS(test):  # ——剔除异常点和PLS模型预测

    #    plt.figure(24)
    #    ymax =38.53
    #    ynir_trhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='U',header = None)
    #    Y0=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='V',header = None)
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),ynir_trhat.iloc[:,0],Y0.iloc[:,0],'oy')
    #    plt.title('标准正态变换和PLSR的训练集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel ('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #
    #    plt.figure(25)
    #    ysnv_testhat=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='W',header = None)
    #    Y0_test=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='X',header = None)
    #    plt.plot(list(range(round(ymax))),list(range(round(ymax))),ysnv_testhat.iloc[:,0],Y0_test.iloc[:,0],'*y')
    #    plt.title('标准正态变换和PLSR的验证集预测结果')
    #    plt.xlabel('预测值（mgN/100g）')
    #    plt.ylabel('真实值（mgN/100g）')
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    #
    #
    #
    #    plt.figure(26)
    #    xrow_snv=107
    #    best_threshold_snv=2.7906
    #    Dis_snv=pd.read_excel(io=(os.path.join(basePath,'The_mid_data.xlsx')),usecols='Y',header = None)
    #    plt.plot(list(range(xrow_snv)),Dis_snv,'bo')
    #    plt.plot(np.arange(1,xrow_snv,0.1),np.repeat(np.array(best_threshold_snv).reshape(1,1),len(np.arange(1,xrow_snv,0.1)),axis=0),'--r',LineWidth=2)
    #    plt.title('标准正态变换的马氏距离和最佳阈值')
    #    plt.xlabel('样品序号')
    #    plt.ylabel('马氏距离')
    #    plt.xlim([0,xrow_snv])  # x轴边界
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()
    ch0 = 84.9204
    ch0_snvte = np.repeat(np.array(ch0).reshape(1, 1), len(test), axis=0)
    xish = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AL', header=None)
    ymsc_testhat = ch0_snvte + np.dot(test, xish)
    return ymsc_testhat


def ax_erase_PLS():  # 剔除异常点和PLS模型预测

    plt.figure(27)
    ymax = 38.53
    yax_trhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='Z', header=None)
    Y0 = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AA', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), yax_trhat.iloc[:, 0], Y0.iloc[:, 0], 'o')
    plt.title('标准化和PLSR的训练集预测结果')
    plt.xlabel('Predictive（mgN/100g）')
    plt.ylabel('True value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    ymax = 34.87
    plt.figure(28)
    yax_testhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AB', header=None)
    Y0_test = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AC', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), yax_testhat.iloc[:, 0], Y0_test.iloc[:, 0], '*')
    plt.title('Validation set prediction results of standardization and PLSR')
    plt.xlabel('Predictive value（mgN/100g）')
    plt.ylabel('True value（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    plt.figure(29)
    xrow_ax = 107
    best_threshold_ax = 2.7906
    Dis_ax = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AD', header=None)
    plt.plot(list(range(xrow_ax)), Dis_ax, 'bo')
    plt.plot(np.arange(1, xrow_ax, 0.1),
             np.repeat(np.array(best_threshold_ax).reshape(1, 1), len(np.arange(1, xrow_ax, 0.1)), axis=0), '--r',
             LineWidth=2)
    plt.title('标准化的马氏距离和最佳阈值')
    plt.xlabel('样品序号')
    plt.ylabel('马氏距离')
    plt.xlim([0, xrow_ax])  # x轴边界
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()


def SG_erase_PLS():  # 剔除异常点和PLS模型预测

    plt.figure(30)
    ymax = 38.53
    ySG_trhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AE', header=None)
    Y0 = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AF', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), ySG_trhat.iloc[:, 0], Y0.iloc[:, 0], 'ok')
    plt.title('SavitZky-Golay卷积平滑法和PLSR的训练集预测结果')
    plt.xlabel('预测值（mgN/100g）')
    plt.ylabel('真实值（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    ymax = 34.87
    plt.figure(31)
    ySG_testhat = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AF', header=None)
    Y0_test = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AG', header=None)
    plt.plot(list(range(round(ymax))), list(range(round(ymax))), ySG_testhat.iloc[:, 0], Y0_test.iloc[:, 0], '*k')
    plt.title('SavitZky-Golay卷积平滑法和PLSR的验证集预测结果')
    plt.xlabel('预测值（mgN/100g）')
    plt.ylabel('真实值（mgN/100g）')
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()

    # %% 作图(马氏距离——最佳阈值图）
    plt.figure(32)
    xrow_SG = 107
    best_threshold_SG = 3.8364
    Dis_SG = pd.read_excel(io=(os.path.join(basePath, 'The_mid_data.xlsx')), usecols='AD', header=None)
    plt.plot(list(range(xrow_SG)), Dis_SG, 'bo')
    plt.plot(np.arange(1, xrow_SG, 0.1),
             np.repeat(np.array(best_threshold_SG).reshape(1, 1), len(np.arange(1, xrow_SG, 0.1)), axis=0), '--r',
             LineWidth=2)
    plt.title('SavitZky-Golay卷积平滑法的马氏距离和最佳阈值')
    plt.xlabel('Sample serial number:')
    plt.ylabel('mahalanobis distance')
    plt.xlim([0, xrow_SG])  # x轴边界
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.show()


#

'''EraseAndPLSpredict001——剔除异常点和PLS模型预测'''


def EraseAndPLSpredict001():
    norm_ysgp = norm_pca(data_ys_train.iloc[:, :-1])  # 原始光谱
    norm_ysgp_msc = norm_pca(data_ysgp_msc.iloc[n[0:n_choose], :])  # 多元散射校正MSC
    norm_ysgp_snv = norm_pca(data_ysgp_snv.iloc[n[0:n_choose], :])  # 标准正态变量交化SNV
    norm_ysgp_Nor = norm_pca(data_ysgp_Nor.iloc[n[0:n_choose], :])  # % 归一化Normalize
    norm_ysgp_MCX = norm_pca(data_ysgp_MCX.iloc[n[0:n_choose], :])  # % 数据中心化(Mean centering)
    norm_ysgp_ax = norm_pca(data_ysgp_ax.iloc[n[0:n_choose], :])  # 标准化Autoscales
    norm_ysgp_nir = norm_pca(data_ysgp_nir.iloc[n[0:n_choose], :])  # (11点)移动窗口平滑/Moving-average method
    norm_ysgp_S_G = norm_pca(data_ysgp_S_G.iloc[n[0:n_choose], :])  # SavitZky一Golay卷积平滑法及求导
    norm_ysgp_S_G1 = norm_pca(data_ysgp_S_G1.iloc[n[0:n_choose], :])  # SavitZky-Golay一阶求导
    norm_ysgp_S_G2 = norm_pca(data_ysgp_S_G2.iloc[n[0:n_choose], :])  # SavitZky-Golay二阶求导
    '''use pca'''
    pca = PCA(n_components=106)
    score_ysgp, latent_ysgp = pca.fit_transform(norm_ysgp), np.array(pca.explained_variance_ratio_).reshape(106, 1)

    score_msc, latent_msc = pca.fit_transform(norm_ysgp_msc), np.array(pca.explained_variance_ratio_).reshape(106, 1)

    score_snv, latent_snv = pca.fit_transform(norm_ysgp_snv), np.array(pca.explained_variance_ratio_).reshape(106, 1)

    score_Nor, latent_Nor = pca.fit_transform(norm_ysgp_Nor), np.array(pca.explained_variance_ratio_).reshape(106, 1)

    #    score_MCX,latent_MCX=pca.fit_transform(norm_ysgp_MCX),np.array(pca.explained_variance_ratio_).reshape(106,1)
    #
    #    score_ax,latent_ax= pca.fit_transform(norm_ysgp_ax),np.array(pca.explained_variance_ratio_).reshape(106,1)
    #
    #    score_nir,latent_nir= pca.fit_transform(norm_ysgp_nir),np.array(pca.explained_variance_ratio_).reshape(106,1)
    #
    #    score_SG,latent_SG=pca.fit_transform(norm_ysgp_S_G),np.array(pca.explained_variance_ratio_).reshape(106,1)
    #
    #    score_SG1,latent_SG1= pca.fit_transform(norm_ysgp_S_G1),np.array(pca.explained_variance_ratio_).reshape(106,1)
    #
    #    score_SG2,latent_SG2= pca.fit_transform(norm_ysgp_S_G2),np.array(pca.explained_variance_ratio_).reshape(106,1)

    # DataScores = [score_ysgp; score_msc; score_snv; score_Nor; score_MCX; score_ax;...
    #                          score_nir; score_SG; score_SG1; score_SG2 ]# 不包含score_d1; score_d2

    DataLatent = np.concatenate(
        [latent_ysgp, latent_msc, latent_snv, latent_Nor, latent_MCX, latent_ax, latent_nir, latent_SG, latent_SG1,
         latent_SG2], axis=1)
    Nmod_data = 10  # 总的所有预处理后数据类型
    percent_explained = np.zeros((DataLatent.shape[0], DataLatent.shape[1]))  # 储存累计主成分所占百分比
    #    for  iNmod in range(Nmod_data):
    #        percent_explained[:,iNmod]= 100*np.cumsum(DataLatent[:,iNmod],axis=0)/sum(DataLatent[:,iNmod])# 累计主成分所占百分比
    #    N_pcm = 20 # 预设置PCA主成分个数
    #
    #    # 画出不同光谱预处理下，PCA处理的各自校正集后的前几项主成分贡献图
    #    X_plot=list(range(1,N_pcm+1))
    #
    #    Color = ['b','r','g','c','m','y','k']
    #    plt.figure(11)
    #    for iNmod in range(7):
    #         color2 = Color[iNmod]  #对应的颜色
    #         plt.plot(X_plot,percent_explained[X_plot,iNmod],color2,linewidth=1)
    #    plt.plot(X_plot,percent_explained[X_plot,7],'-mo' ,linewidth=1)
    #    plt.plot(X_plot,percent_explained[X_plot,8],'-ro',LineWidth=1)
    #    plt.plot(X_plot,percent_explained[X_plot,9],'-bo',LineWidth=1)
    #    plt.yticks(list(range(1,100,3)))  # 设置y刻度
    #    plt.xticks(list(range(0,21,2)))  # 设置y刻度
    #    plt.title('不同光谱预处理的主成份分析')
    #    plt.xlabel('主成分数')
    #    plt.ylabel('贡献率 %')
    #    plt.legend(['原始','MSC','SNV','Normalize','Mean centering','Autoscales','移动窗口平滑','Savitzky-Golay卷积平滑','Savitzky-Golay一阶求导','Savitzky-Golay二阶求导'])
    #    matplotlib.rcParams['axes.unicode_minus'] =False#用于显示负号
    #    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    #    plt.show()

    NewScore_msc = score_msc[:, 0:6]  # 多元散射校正MSC
    NewScore_Nor = score_Nor[:, 0:4]  # 归一化Normalize
    #    NewScore_MCX = score_MCX[:,0:3]#数据中心化(Mean centering)
    #    NewScore_nir = score_nir[:,0:2]# (11点)移动窗口平滑/Moving-average method
    NewScore_snv = score_snv[:, 0:6]  # 标准正态变量交化SNV
    #    NewScore_ax = score_ax[:,0:6]# 标准化Autoscales
    #    NewScore_SG = score_SG[:,0:2]# % SavitZky一Golay卷积平滑法及求导
    #    NewScore_SG1 = score_SG1[:,0:20]#  % SavitZky-Golay一阶求导
    #    NewScore_SG2 = score_SG2[:,0:20]#  % SavitZky-Golay二阶求导
    #

    lmd_msc = latent_msc[0:6].T  # ; % 多元散射校正MSC的前2个主成分对应的方差在lmd_msc向量中
    lmd_Nor = latent_Nor[0:4].T  # % 归一化Normalize的前2个主成分对应的方差在lmd_Nor向量中
    #    lmd_MCX = latent_MCX[0:3].T# % 数据中心化MCXmalize的前2个主成分对应的方差在lmd_MCX向量中
    #    lmd_nir = latent_nir[0:2].T# % (11点)移动窗口平滑nir的前2个主成分对应的方差在lmd_nir向量中
    lmd_snv = latent_snv[0:6].T  # % 标准正态变量交化SNV
    #    lmd_ax = latent_ax[0:6].T# % 标准化Autoscales
    #    lmd_SG = latent_SG[0:2].T# SavitZky一Golay卷积平滑法及求导
    #    lmd_SG1 = latent_SG1[0:20].T#  % SavitZky-Golay一阶求导
    #    lmd_SG2 = latent_SG2[0:20].T# SavitZky-Golay二阶求导
    #

    data_msc_train = np.concatenate(
        [data_ysgp_msc.iloc[n[0:n_choose], :], np.array(data_ys_train.iloc[:, -1]).reshape(107, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据
    data_msc_test = np.concatenate(
        [data_ysgp_msc.iloc[n[n_choose:], :], np.array(data_ys_test.iloc[:, -1]).reshape(36, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据
    data_Nor_train = np.concatenate(
        [data_ysgp_Nor.iloc[n[0:n_choose], :], np.array(data_ys_train.iloc[:, -1]).reshape(107, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据
    data_Nor_test = np.concatenate(
        [data_ysgp_Nor.iloc[n[n_choose:], :], np.array(data_ys_test.iloc[:, -1]).reshape(36, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据
    #    data_nir_train = np.concatenate([data_ysgp_nir.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
    #    data_nir_test = np.concatenate([data_ysgp_nir.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
    #    data_MCX_train = np.concatenate([data_ysgp_MCX.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
    #    data_MCX_test = np.concatenate([data_ysgp_MCX.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
    data_snv_train = np.concatenate(
        [data_ysgp_snv.iloc[n[0:n_choose], :], np.array(data_ys_train.iloc[:, -1]).reshape(107, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据
    data_snv_test = np.concatenate(
        [data_ysgp_snv.iloc[n[n_choose:], :], np.array(data_ys_test.iloc[:, -1]).reshape(36, 1)],
        axis=1)  # % 多元散射校正MSC的校正数据


#    data_ax_train = np.concatenate([data_ysgp_ax.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_ax_test = np.concatenate([data_ysgp_ax.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG_train = np.concatenate([data_ysgp_S_G.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG_test = np.concatenate([data_ysgp_S_G.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG1_train = np.concatenate([data_ysgp_S_G1.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG1_test = np.concatenate([data_ysgp_S_G1.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG2_train = np.concatenate([data_ysgp_S_G2.iloc[n[0:n_choose],:],np.array(data_ys_train.iloc[:,-1]).reshape(107,1)],axis=1)# % 多元散射校正MSC的校正数据
#    data_SG2_test = np.concatenate([data_ysgp_S_G2.iloc[n[n_choose:],:],np.array(data_ys_test.iloc[:,-1]).reshape(36,1)],axis=1)# % 多元散射校正MSC的校正数据
#    
#    '''调制阈值权重，剔除异常点'''
#    Weight =np.array((np.arange(0.1,4,0.25)))
# 多元散射校正MSC的最佳异常点剔除
# 不断调整权重，获取最佳的PLS的RMSEC_MSC情况下的最佳weight，PLS的建模参数，剔除掉异常点后的数据
#    MSC_erase_PLS()
#    Nor_erase_PLS()
#    MCX_erase_PLS()
#    nir_erase_PLS()
#    snv_erase_PLS()
#    ax_erase_PLS()
#    SG_erase_PLS()


def process(file_path):
    data_ys = pd.DataFrame(pd.read_excel(io=(os.path.join(basePath, 'spectral&tvbn_content.xlsx')), usecols='A:HT',
                                         header=None))  # 读取原始数据(光谱数据和对应的TVB-N)
    data_ysrow = data_ys.shape[0]
    data_yscol = data_ys.shape[1]  # 数据的行和列
    data_ysbc = pd.DataFrame(
        pd.read_excel(io=(os.path.join(basePath, 'wavelength.xlsx')), header=None))  # 光谱的波长范围（横坐标数据）
    data_ystvbn = data_ys.iloc[:, data_yscol - 1]  # 测量的光谱曲线对应的产品TVB-N含量（目标）
    data_ysgp = data_ys.iloc[:, :]  # 光谱数据，每一行数据可以绘出一条光谱曲线（纵坐标数据）

    ''' 消除光谱噪声'''
    # data_pretreatment(data_ysbc,data_ysgp)
    # EraseAndPLSpredict001()

    '''以下为pls预测部分'''
    test_data = pd.DataFrame(pd.read_excel(io=file_path, header=None))  # 读取原始数据(光谱数据和对应的TVB-N)
    test_data_mean = np.mean(data_ysgp)  # 表示所有样品的光谱在各个波长点处求平均值
    ymsc_data = msc(test_data, test_data_mean)
    yNor_data = normaliz(test_data)
    ysnv_data = data_ysgp_snv = snv(test_data)
    ymsc_testhat = MSC_erase_PLS(ymsc_data)
    # yNor_testhat=Nor_erase_PLS(yNor_data)
    # ysnv_testhat=snv_erase_PLS(ysnv_data)
    print('MSC_erase_PLS预测结果为：', ymsc_testhat)

    data_ysgp = test_data.iloc[:, :]  # 光谱数据，每一行数据可以绘出一条光谱曲线（纵坐标数据）

    iNys_material = test_data.shape[0]
    Color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 绘图的颜色

    for i in range(iNys_material):
        plt.plot(list(data_ysbc.iloc[0, :]), list(data_ysgp.iloc[i, :]), color=Color[random.randint(0, 6)],
                 linewidth=0.5)

    plt.title('Original spectrogram')
    plt.xlabel('wavelength/nm Wavelength')
    plt.ylabel('reflectivity')

    plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False

    workpath = os.getcwd()
    print(workpath)
    plt.savefig(workpath + "/static/result/result.png")

    return str(ymsc_testhat)
