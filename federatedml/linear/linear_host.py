import numpy as np

from baseCommon.plogger import *
from federatedml.linear.linear_basic import linear_basic
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

file = "F:\\datasource\\linr_data_x.csv"


class linear_host(linear_basic):
    file = "F:\\datasource\\linr_data_x.csv"

    uuid = None
    alldata = None  # linear_host.public_key = None
    data = None
    traindata = None
    testdata = None
    w = None
    '''
    输入数据
    '''

    def loaddata(file):
        df = pd.read_csv(file, skiprows=0)
        pp = df.values
        print("KKK=", pp[:2, :].tolist())
        #
        pp = np.hstack((df, np.ones((np.shape(df)[0], 1))))  # 给x添加x[0]项

        linear_host.data = pp
        linear_host.alldata = pp
        return linear_host.alldata

    #
    # def linr_GetPK(self,tradecode,uuid,reqdata):
    #     LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
    #     if tradecode == 'linr_pksender':
    #         linear_host.uuid = uuid
    #         linear_host.public_key = reqdata
    #         req = "get it!"
    #         return 0, req
    #     else:
    #         LOGGER.info("linr_GeneratePK Error!")
    #         return 500, "linr_GeneratePK Error!"
    '''模型保存啊'''

    def model_save(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "model_save":
            name, path = reqdata
            np.save(path + name + "_host", linear_host.w)
            return 0, reqdata
        else:
            LOGGER.info("model_save Error!")
            return 500, "model_save Error!"

    '''这是模型测试'''

    def test_wx_H(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "test_wx_H":
            name, path, dataname, datapath = reqdata
            w = np.load(path + name + "_host.npy")
            data = np.load(datapath + dataname + '_host.npz')

            wx_G = np.matmul(data['arr_1'][:, 1:], w)

            return 0, wx_G
        else:
            LOGGER.info("test_wx_H Error!")
            return 500, "test_wx_H Error!"

    def train_test_div(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "train_test_div":
            path, trainpart = reqdata
            linear_host.traindata, linear_host.testdata = linear_basic.train_test(linear_host.alldata, trainpart)
            print(path)
            np.savez(path + '_host.npz', linear_host.traindata, linear_host.testdata)
            linear_host.data = linear_host.traindata
            return 0, reqdata
        else:
            LOGGER.info("train_test_div Error!")
            return 500, "train_test_div Error!"

    ## 定义
    def shuffle_data(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "shuffle_data":
            linear_host.loaddata(linear_host.file)
            linear_host.uuid = uuid
            linear_host.alldata = linear_host.alldata[reqdata]  # reqdata 就是 indexs
            return 0, reqdata
        else:
            LOGGER.info("shuffle_data Error!")
            return 500, "shuffle_data Error!"

    ## 定义
    def set_index_data(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "linear_random_index_array":

            linear_host.uuid = uuid
            linear_host.data = linear_basic.get_index_array(linear_host.traindata, 1, reqdata)  # reqdata 就是 indexs

            return 0, reqdata
        else:
            LOGGER.info("linear_random_index_array Error!")
            return 500, "linear_random_index_array Error!"

    '''
    生成初始化的wx，并转化成list'''

    @classmethod
    def generate_wx(cls):
        dd = linear_host.data
        x = dd[:, 1:]
        linear_host.w = linear_basic.init_w(x)
        wx_H = linear_basic.wx(x, linear_host.w)

        # wx_H = wx_H.tolist()
        wx_H = list(map(lambda x: x[0], wx_H.tolist()))
        return wx_H

    '''
    grpc，server端，将wx返回'''

    def remote_wx(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "linear_remote_wx":

            linear_host.uuid = uuid
            wx = linear_host.generate_wx()

            return 0, wx
        else:
            LOGGER.info("remote_wx Error!")
            return 500, "remote_wx Error!"

    '''
        grpc，server端，接受加密的残差residual，返回加密的梯度div_j_H'''

    def get_residual(tradecode, uuid, reqdata):

        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "linear_get_residual":

            linear_host.uuid = uuid
            residual = reqdata  # 拿到加密d
            m = len(residual)
            residual = np.array(residual).reshape(m, 1)
            x = linear_host.data[:, 1:]  # 明文特征矩阵
            div_j_H = linear_basic.gradient(residual, x)  # 加密的梯度
            div_j_H = list(map(lambda x: x[0], div_j_H.tolist()))

            return 0, div_j_H
        else:
            LOGGER.info("get_residual Error!")
            return 500, "get_residual Error!"

    '''
        grpc，server端，接收解密后的梯度div_j_H，返回更新后的wx值'''

    def get_div_j(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "linear_req_div_j":
            linear_host.uuid = uuid
            div_j_H = reqdata[0]  # 拿到加密梯度
            m = len(div_j_H)
            div_j_H = np.array(div_j_H).reshape(m, 1)
            alpha = reqdata[1]
            x = linear_host.data[:, 1:]  # 明文特征矩阵
            linear_host.w = linear_basic.update_w(linear_host.w, div_j_H, alpha)
            print(linear_host.w)
            wx_H = linear_basic.wx(x, linear_host.w)
            wx_H = list(map(lambda x: x[0], wx_H.tolist()))

            return 0, wx_H

        else:
            LOGGER.info("linr_req_div_j Error!")
            return 500, "linr_req_div_j Error!"


if __name__ == '__main__':
    lh = linear_host()
    wx = lh.generate_wx(file)
