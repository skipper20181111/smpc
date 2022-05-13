import numpy as np

from baseCommon.plogger import *
from federatedml.logistic.logistic_basic import logistic_basic
# from secureprotol.encrypt import PaillierEncrypt
from baseCommon.logger import LogClass, ON
import pandas as pd
import multiprocessing as mp

# logclass=LogClass("testlinrlog.txt")
# LOGGER = logclass.get_Logger("levelname!!", ON.DEBUG)


LoggerFactory.set_directory(directory="./")
LOGGER = getLogger()

# interdata = [1,2]#根据求交结果导入
# data = interdata

file = r'f:/onlyx.csv'


class logistic_host(logistic_basic):
    file = r'f:/onlyx.csv'

    uuid = None
    alldata = None
    data = None
    traindata = None
    testdata = None
    w = None
    '''
    输入数据
    '''

    def loaddata(file):
        df = pd.read_csv(file)
        pp = df.values
        #
        pp = np.hstack((df, np.ones((np.shape(df)[0], 1))))  # 给x添加x[0]项

        logistic_host.data = pp
        logistic_host.alldata = logistic_host.data
        return logistic_host.alldata

    '''模型保存啊'''

    def logistic_model_save(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_model_save":
            name, path = reqdata
            np.save(path + name + "_host", logistic_host.w, allow_pickle=True)
            return 0, reqdata
        else:
            LOGGER.info("logistic_model_save Error!")
            return 500, "logistic_model_save Error!"

    '''这是模型测试'''

    def logistic_test_wx_H(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_test_wx_H":

            name, path, dataname, datapath = reqdata
            print('wo')
            w = np.load(path + name + "_host.npy")
            print('wo1')
            data = np.load(datapath + dataname + '_host.npz')
            print('wo2')
            wx_G = np.matmul(data['arr_1'][:, 1:], w)

            return 0, wx_G
        else:
            LOGGER.info("logistic_test_wx_H Error!")
            return 500, "logistic_test_wx_H Error!"

    def logistic_train_test_div(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_train_test_div":
            datapath_name, train_part = reqdata
            logistic_host.traindata, logistic_host.testdata = logistic_basic.train_test(logistic_host.alldata,
                                                                                        train_part)

            np.savez(datapath_name + '_host.npz', logistic_host.traindata, logistic_host.testdata)
            logistic_host.data = logistic_host.traindata
            return 0, reqdata
        else:
            LOGGER.info("logistic_train_test_div Error!")
            return 500, "logistic_train_test_div Error!"

    ## 定义
    def logistic_shuffle_data(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_shuffle_data":
            logistic_host.loaddata(logistic_host.file)
            logistic_host.uuid = uuid
            logistic_host.alldata = logistic_host.alldata[reqdata]  # reqdata 就是 indexs
            return 0, reqdata
        else:
            LOGGER.info("logistic_ Error!")
            return 500, "logistic_ Error!"

    ## 定义
    def logistic_random_index_array(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_random_index_array":

            logistic_host.uuid = uuid
            logistic_host.data = logistic_host.get_index_array(logistic_host.traindata, 1, reqdata)  # reqdata 就是 indexs

            return 0, reqdata
        else:
            LOGGER.info("logistic_random_index_array Error!")
            return 500, "logistic_random_index_array Error!"

    '''
    生成初始化的wx，并转化成list'''

    @classmethod
    def generate_wx(cls):
        dd = logistic_host.data
        x = dd[:, 1:]
        logistic_host.w = logistic_basic.init_w(x)
        wx_H = logistic_basic.wx(x, logistic_host.w)

        # wx_H = wx_H.tolist()
        wx_H = list(map(lambda x: x[0], wx_H.tolist()))
        return wx_H

    '''
    grpc，server端，将wx返回'''

    def remote_wx(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_remote_wx":

            logistic_host.uuid = uuid
            wx = logistic_host.generate_wx()

            return 0, wx
        else:
            LOGGER.info("logistic_remote_wx Error!")
            return 500, "logistic_remote_wx Error!"

    '''
        grpc，server端，接受加密的残差residual，返回加密的梯度div_j_H'''

    def get_residual(tradecode, uuid, reqdata):

        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_get_residual":

            logistic_host.uuid = uuid
            residual = reqdata  # 拿到加密d
            m = len(residual)
            residual = np.array(residual).reshape(m, 1)
            x = logistic_host.data[:, 1:]  # 明文特征矩阵
            div_j_H = logistic_basic.gradient(residual, x)  # 加密的梯度
            div_j_H = list(map(lambda x: x[0], div_j_H.tolist()))

            return 0, div_j_H
        else:
            LOGGER.info("logistic_get_residual Error!")
            return 500, "logistic_get_residual Error!"

    '''
        grpc，server端，接收解密后的梯度div_j_H，返回更新后的wx值'''

    def get_div_j(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_req_div_j":
            logistic_host.uuid = uuid
            div_j_H = reqdata[0]  # 拿到加密梯度
            m = len(div_j_H)
            div_j_H = np.array(div_j_H).reshape(m, 1)
            alpha = reqdata[1]
            x = logistic_host.data[:, 1:]  # 明文特征矩阵
            logistic_host.w = logistic_basic.update_w(logistic_host.w, div_j_H, alpha)
            print(logistic_host.w)
            wx_H = logistic_basic.wx(x, logistic_host.w)
            wx_H = list(map(lambda x: x[0], wx_H.tolist()))

            return 0, wx_H

        else:
            LOGGER.info("logistic_req_div_j Error!")
            return 500, "logistic_req_div_j Error!"


if __name__ == '__main__':
    lh = logistic_host()
    wx = lh.generate_wx(file)


def linear_host():
    return None
