import numpy as np

from baseCommon.plogger import *
from federatedml.logistic.logistic_basic import logistic_basic
from secureprotol.encrypt import PaillierEncrypt
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


class logistic_host(logistic_basic, PaillierEncrypt):
    file = "F:\\datasource\\linr_data_x.csv"

    def __init__(self):
        super().__init__()
        uuid = None
        # logistic_host.public_key = None
        data = None
        w = None

    '''
    输入数据
    '''

    def loaddata(file):
        df = pd.read_csv(file)
        data = df.values
        logistic_host.data = data
        return logistic_host.data

    #
    # def linr_GetPK(self,tradecode,uuid,reqdata):
    #     LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
    #     if tradecode == 'linr_pksender':
    #         logistic_host.uuid = uuid
    #         logistic_host.public_key = reqdata
    #         req = "get it!"
    #         return 0, req
    #     else:
    #         LOGGER.info("linr_GeneratePK Error!")
    #         return 500, "linr_GeneratePK Error!"

    '''
    生成初始化的wx，并转化成list'''

    def generate_wx():
        dd = logistic_host.loaddata(file)
        x = dd[:, 1:]
        logistic_host.w = logistic_basic.init_w(x)
        wx_H = logistic_basic.wx(x, logistic_host.w)
        print(len(wx_H))
        # wx_H = wx_H.tolist()
        wx_H = list(map(lambda x: x[0], wx_H.tolist()))

        # print(wx_H)
        # print("pkpkpkpk",logistic_host.public_key)
        # wx_HE = list(map(lambda x:logistic_host.public_key.encrypt(x[0]),wx_HE))
        print(len(wx_H))

        # print("wwwwwhhhh",wx_H)
        # wx_H = PaillierEncrypt.encrypt(wx_H)#加密wx_H
        # print("wx_HE",wx_H)
        return wx_H

    '''
    grpc，server端，将wx返回'''

    def remote_wx(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_remote_wx":
            # print(reqdata)
            logistic_host.uuid = uuid
            wx = logistic_host.generate_wx()
            # print("hhhhhhh=",wx)
            # print(wx)
            return 0, wx
        else:
            LOGGER.info("remote_wx Error!")
            return 500, "remote_wx Error!"

    #    LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
    #    if tradecode == "logistic_remote_wx":
    #        # print(reqdata)
    #        logistic_host.uuid = uuid
    #        wx = logistic_host.generate_wx()
    #        # print("hhhhhhh=",wx)
    #        # print(wx)
    #        return 0, wx
    #    else:
    #        LOGGER.info("remote_wx Error!")
    #        return 500, "remote_wx Error!"

    '''
        grpc，server端，接受加密的残差residual，返回加密的梯度div_j_H'''

    def get_residual(tradecode, uuid, reqdata):

        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_get_residual":
            # print("hhhhhhhhhhh=",reqdata)
            logistic_host.uuid = uuid
            residual = reqdata  # 拿到加密d
            m = len(residual)
            residual = np.array(residual).reshape(m, 1)

            # print(np.shape(wx_H))
            x = logistic_host.data[:, 1:]  # 明文特征矩阵
            # print(np.shape(x))
            div_j_H = logistic_basic.gradient(residual, x)  # 加密的梯度
            div_j_H = list(map(lambda x: x[0], div_j_H.tolist()))
            # print(div_j_H)
            return 0, div_j_H
        else:
            LOGGER.info("get_residual Error!")
            return 500, "get_residual Error!"

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
            # print(np.shape(logistic_host.w))
            # print("newWWWW",logistic_host.w)
            wx_H = logistic_basic.wx(x, logistic_host.w)
            wx_H = list(map(lambda x: x[0], wx_H.tolist()))

            return 0, wx_H
            # print("hhhhhh",wx_H)
        else:
            LOGGER.info("linr_req_div_j Error!")
            return 500, "linr_req_div_j Error!"


if __name__ == '__main__':
    lh = logistic_host()
    wx = lh.generate_wx(file)
    # print(wx)
