import numpy as np
import random
import os
import pickle

'''
G端：任务发起方，拥有x值和y值
H端：联合训练方，拥有x值
'''

from baseCommon.httpcall import Httpapi

'''
调用java端接口，根据dataName得到filePath
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
计算IV值的data进行文件名转化，转化为原始文件名
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


'''
G端：任务发起方，拥有x值和y值
H端：联合训练方，拥有x值
'''


def mean_squared_error(y, y_predict):
    m = np.shape(y)[0]
    return sum((y - y_predict) ** 2) / m


def root_mean_squared_error(y, y_predict):
    return mean_squared_error(y, y_predict) ** 0.5


def formatstr(number):
    return '%.6f' % number


def mean_absolute_error(y, y_predict):
    m = np.shape(y)[0]
    return sum(((y - y_predict) ** 2) ** 0.5) / m


def r_squared(y, y_predict):
    return 1 - (mean_squared_error(y, y_predict) / np.var(y))  ##就是一个r2而已啦，洒洒水啦


def model_save(model_name, parameter):
    np.save(model_name, parameter)


class linear_basic():
    '''
    初始化w，入参x    为x的矩阵(G,H两端都需要)(行数数据条数，列数是特征数+1)
    出参init_w为单列矩阵
    '''

    def init_w(x):
        init_w = np.random.rand(np.shape(x)[1], 1)
        return init_w

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

    def train_test(all_data, part):
        m, n = np.shape(all_data)
        div_num = int(m * part)
        if part > 0 and part < 1:
            return all_data[:div_num, :], all_data[div_num:, :]
        else:
            return all_data, all_data

    def get_index_array(data, flag, indexs):
        if flag == 0:
            return data
        else:
            return data[indexs]

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
        residual = wx_G + wx_H - y
        return residual

    '''
    梯度，输入为残差d(一列矩阵)和x(多维矩阵)
    '''

    def gradient(residual, x):
        div_j = ((np.matmul(np.transpose(x), residual)) / (np.shape(x)[0] * np.shape(x)[1]))
        return div_j

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
收敛条件'''


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
            print('#####################################################')
            print(converged.num)
        if converged.num >= 10:
            return True
        return False

    def jtheta(d):
        return sum(d ** 2)
