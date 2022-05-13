import numpy as np
import random
import os
import pickle
from baseCommon.pymysqlclass import *
'''
G端：任务发起方，拥有x值和y值
H端：联合训练方，拥有x值
'''

from baseCommon.httpcall import Httpapi
'''
每一次产生的梯度伴随着大量白噪声
为了减弱这种白噪声的影响，需要对其求均值。

'''
def get_mean(old,new,mean_of_what):
    partition=1/mean_of_what
    return partition*new + (1-partition)*old
'''
利用测试数据集的标签列y_test来测试模型效果，生成诸如roc曲线，ks曲线等的评判标准文件
'''
def model_evaluate(wx_H,wx_G,y_test):
    pointnum = 50
    roclist = []  ### 用循环计算每一个roc曲线的点，包括两个值，x与y坐标

    kslist = []  ### 用循环计算每一个ks曲线的点，包括三条线，因此有四个值，三个y坐标，一个x坐标

    for i in range(pointnum):
        y_pre = np.array(logistic_basic.sigmoid(wx_H + wx_G) >= ((i) / pointnum), dtype=np.int32)
        yf = sum(y_test == 0)[0]
        yc = sum(y_test == 1)[0]
        tn = sum((y_pre[y_test == 0]) == 0)
        fp = sum((y_pre[y_test == 0]) == 1)
        fn = sum((y_pre[y_test == 1]) == 0)
        tp = sum((y_pre[y_test == 1]) == 1)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        roclist.append([fpr, tpr])
        kslist.append([tpr, fpr, tpr - fpr, i / pointnum])
    ksarray = np.array(kslist)
    rocarray = np.array(roclist)
    ksvalue = max(ksarray[:, -1])
    auclist = []  ### 利用微积分计算auc的数值
    for i in range(pointnum - 1):
        dfpr = rocarray[i, 0] - rocarray[i + 1, 0]
        meandtpr = (rocarray[i, 1] + rocarray[i + 1, 1]) / 2
        auclist.append(dfpr * meandtpr)
    auc = sum(auclist)

    '''这个是为了生成string形式的roc与ks列表'''
    roclist = []  ### 用循环计算每一个roc曲线的点，包括两个值，x与y坐标

    kslist = []  ### 用循环计算每一个ks曲线的点，包括三条线，因此有四个值，三个y坐标，一个x坐标

    for i in range(pointnum):
        y_pre = np.array(logistic_basic.sigmoid(wx_H + wx_G) >= ((i) / pointnum), dtype=np.int32)
        yf = sum(y_test == 0)[0]
        yc = sum(y_test == 1)[0]
        tn = sum((y_pre[y_test == 0]) == 0)
        fp = sum((y_pre[y_test == 0]) == 1)
        fn = sum((y_pre[y_test == 1]) == 0)
        tp = sum((y_pre[y_test == 1]) == 1)
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        roclist.append([formatstr(fpr), formatstr(tpr)])
        kslist.append([formatstr(tpr), formatstr(fpr), formatstr(tpr - fpr), formatstr(i / pointnum)])

    y_pre = np.array(logistic_basic.sigmoid(wx_H + wx_G) >= 0.5, dtype=np.int32)  ###计算0.5作为分类阈值时的y_pre

    yf = sum(y_test == 0)[0]
    yc = sum(y_test == 1)[0]
    tn = sum((y_pre[y_test == 0]) == 0)
    fp = sum((y_pre[y_test == 0]) == 1)
    fn = sum((y_pre[y_test == 1]) == 0)
    tp = sum((y_pre[y_test == 1]) == 1)
    recall = tp / (tp + fn)
    pricision = tp / (tp + fp)
    return {'tp': formatstr(tp),
            'tn': formatstr(tn),
            'fn': formatstr(fn),
            'fp': formatstr(fp),
            'tpr': formatstr(tp / (tp + fn)),
            'fpr': formatstr(fp / (fp + tn)),
            'ppv': formatstr(tp / (tp + fp)),
            'npv': formatstr(tn / (tn + fn)),
            'acc': formatstr((tp + tn) / (tp + tn + fp + fn)),
            'f1': formatstr(2 * recall * pricision / (recall + pricision)),
            'roc': roclist,
            'auc': formatstr(auc),
            'ks': kslist,
            'ksvalue': formatstr(ksvalue)}  # 训练模型
'''
将状态码写入数据库
'''
def mysql_iv_model_status(status,jobid):
    sql = 'update smpc_model set model_status=%s where id=%s '
    sqlc = mysqlClass()
    if status=="failed":
        param=['99', jobid]
    elif status=="start":
        param = ['05', jobid]
    elif status=="finished":
        param = ['06', jobid]
    else:
        param = ['99', jobid]
    sqlc._execute(sql, param=param)
'''
下面两个函数利用pickle库将对象进行序列化的保存与读取
'''

def pickle_save(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()

def pickle_load(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

'''
从数据库中读取文件的路径，替换之前的http请求。
'''
def get_path_from_db(filename):
    sqlc = mysqlClass()
    param=[filename]
    sql = "select file_path from smpc_data_info where data_name = %s"
    filepath = sqlc._fetchone(sql, param=param)[0]
    return filepath

'''zmh_2021/9/10'''


def getpath_by_jobid(jobid, method):
    path = r"/data/zhanghui/woe/" + jobid + method + '.txt'
    return path


def get_path_by_name(name):
    http = Httpapi()
    response = http.http_post(url='/smpc/api/dataInfo/getFilePathByDataName', body={"dataName": name})
    path = response["result"]["filePath"]
    return path


'''
计算IV值的data进行文件名转化，转化为原始文件名
@param dataname
@return filename
'''


def inter_dataname_TANS(dataname):
    filename = dataname[:-10]
    return filename


'''
训练的data进行文件名转化，转化为原始文件名
@param dataname
@return filename
'''


def woe_dataname_TANS(dataname):
    filename = dataname[:-14]
    return filename


def model_save_upgrade(parameter, jobid, method):
    save_path = getpath_by_jobid(jobid, method)
    pickle_save(save_path, parameter)


'''
拼路径
'''


def concat_path(jobid, filepath):
    job_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), jobid)
    return job_path


'''
拼文件地址的绝对路径
@param dataname
@param jobid
@param filepath
@return job_filepath
'''


def concat_filepath(dataname, jobid, filepath):
    job_filepath = os.path.join(os.path.dirname(os.path.dirname(filepath)), jobid, dataname)
    return job_filepath


def save_controll_data(controll_data, jobid, method):
    save_path = getpath_by_jobid(jobid, method)
    pickle_save(save_path, controll_data)


def model_save(model_name, parameter):
    np.save(model_name, parameter)


def formatstr(number):
    return str('%.6f' % number)


class logistic_basic():
    '''
    初始化w，入参x    为x的矩阵(G,H两端都需要)(行数数据条数，列数是特征数+1)
    出参init_w为单列矩阵
    '''

    def init_w(x):
        init_w = np.random.rand(np.shape(x)[1], 1)
        return init_w
    '''
    为模型加入正则化项
    '''
    def regularization_model(model, theta, delta):
        if model == 'L2':
            #         print('L1正则化')
            return theta * delta
        elif model == 'L1':
            #         print('L2正则化')
            return delta * np.sign(theta)
        elif model == 'no':
            return np.zeros((np.shape(theta)))
        else:
            #         print('默认L2正则化')
            return theta * delta
    '''
    显而易见，
    '''
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    '''
    用于选取每一次训练所需要的batch数据，也可以用于数据切分为训练集和测试集。
    '''
    def get_random_sample(sample, part):
        m, y = np.shape(sample)
        if part <= m or part >= 1:
            indexs = random.sample(range(m), int(part))
            return 1, indexs
        else:
            return 0, [[-1], [-1]]

    def get_random_sample(sample, part):
        m, y = np.shape(sample)
        if part <= m and part >= 1:
            indexs = random.sample(range(m), int(part))
            return 1, indexs
        else:
            return 0, [[-1], [-1]]
    '''
    分割训练集与测试集
    '''
    def train_test(all, part):
        m, n = np.shape(all)
        div_num = int(m * part)
        if part > 0 and part < 1:
            return all[:div_num, :], all[div_num:, :]
        else:
            return all, all
    '''
    返回数据子集在总数据中的index
    '''
    def get_index_array(sample, flag, indexs):
        if flag == 0:
            return sample
        else:
            return sample[indexs]

    '''
    wx值(单方预测值），（host端需要传递到guest端，根据两边wx之和算出残差）
    入参 x：x（特征）的矩阵
    入参w:w（上一次更新后的w值(单列矩阵)，如果第一次开始则为初始化的w值
    出参 wx:与x行数相等的一列
    '''

    def wx(x, w):
        wx = np.matmul(x, w)
        return wx

    '''
    残差，由G端计算产生，并传到H端
    入参：G_wx，G_wx，分别为两边的wx值,y为G端y值，参数数据格式全部为数据条数长的的单列矩阵
    出参residual：数据形式同入参
    '''

    def compute_d(wx_G, wx_H, y):
        residual = logistic_basic.sigmoid(wx_G + wx_H) - y
        return residual

    '''
    梯度，输入为残差d(一列矩阵)和x(多维矩阵)
    '''

    def gradient(residual, x):
        div_j = ((np.matmul(np.transpose(x), residual)) / (np.shape(x)[0] * np.shape(x)[1]))
        return div_j

    def gradient_regul(x, residual, regul_model, delta, theta, m):
        return ((np.matmul(np.transpose(x), residual)) / (m)) + logistic_basic.regularization_model(regul_model, theta,
                                                                                                    delta) / m

    '''
    更新w值
    入参：旧的w，一列矩阵（长度x的列数）
    入参：div_j：一列矩阵（长度x的列数）
    学习率
    出参：更新后的w
    '''

    def update_w(w, div_j, alpha):
        w = w - alpha * div_j
        return w

    # 计算损失函数 d是残差
    def jtheta(d):
        return sum(d ** 2)


'''
收敛条件
此函数用于逻辑回归的早停策略
逻辑在于，模型的参数不再随时间变动，就代表模型已经收敛了

'''


class converged():
    # def __init__(self):
    mean_of_what = 10
    m = 0
    m_0 = 0
    num = 0
    threshold = 0.1

    def set_threshold(hold):
        converged.threshold = hold

    def init_orign():
        converged.mean_of_what = 10
        converged.m = 0
        converged.m_0 = 0
        converged.num = 0

    def mean_10(loss):
        converged.m_0 = converged.m
        converged.m = (1 - 1 / converged.mean_of_what) * converged.m + (
                1 / converged.mean_of_what) * loss

    def set_period(period):
        converged.mean_of_what = period

    def stop_or_ahead():
        if abs((converged.m_0 / converged.m) - 1) <= converged.threshold:
            converged.num += 1
        if converged.num >= 10:
            return True
        return False

    def jtheta(d):
        return sum(d ** 2)

    def param_ratio(w):
        return w / w[0]

    def logistic_mean_10(standerd_w):
        converged.m_0 = converged.m
        converged.m = (1 - 1 / converged.mean_of_what) * converged.m + (
                1 / converged.mean_of_what) * standerd_w

    def logistic_stop_or_ahead():
        if (sum(abs((converged.m - converged.m_0) / converged.m)) / np.shape(converged.m)[0]) <= converged.threshold:
            print('========================================')
            print(abs((converged.m - converged.m_0) / converged.m))
            converged.num += 1
        if converged.num >= 10:
            return True
        return False


# if __name__ == '__main__':
#     a=get_path_from_db(filename)
#     print()