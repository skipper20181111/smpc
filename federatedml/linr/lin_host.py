import numpy as np
from federatedml.linr.lin_basic import lin_basic
from secureprotol.encrypt import PaillierEncrypt
from baseCommon.logger import LogClass, ON
import pandas as pd
import multiprocessing as mp

logclass = LogClass("testlinrlog.txt")
LOGGER = logclass.get_Logger("levelname!!", ON.DEBUG)

# interdata = [1,2]#根据求交结果导入
# data = interdata

file = "F:\\datasource\\linr_data_x.csv"


class lin_host(lin_basic, PaillierEncrypt):
    file = "F:\\datasource\\linr_data_x.csv"

    def __init__(self):
        super().__init__()
        self.uuid = None
        # self.public_key = None
        self.data = None
        self.w = None

    '''
    输入数据
    '''

    def loaddata(self, file):
        df = pd.read_csv(file)
        data = df.values
        self.data = data
        return self.data

    #
    # def linr_GetPK(self,tradecode,uuid,reqdata):
    #     LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
    #     if tradecode == 'linr_pksender':
    #         self.uuid = uuid
    #         self.public_key = reqdata
    #         req = "get it!"
    #         return 0, req
    #     else:
    #         LOGGER.info("linr_GeneratePK Error!")
    #         return 500, "linr_GeneratePK Error!"

    '''
    生成初始化的wx，并转化成list'''

    def generate_wx(self):
        dd = self.loaddata(file)
        x = dd[:, 1:]
        self.w = lin_basic.init_w(x)
        wx_H = lin_basic.wx(x, self.w)
        print(len(wx_H))
        # wx_H = wx_H.tolist()
        wx_H = list(map(lambda x: x[0], wx_H.tolist()))

        # print(wx_H)
        # print("pkpkpkpk",self.public_key)
        # wx_HE = list(map(lambda x:self.public_key.encrypt(x[0]),wx_HE))
        print(len(wx_H))

        # print("wwwwwhhhh",wx_H)
        # wx_H = PaillierEncrypt.encrypt(wx_H)#加密wx_H
        # print("wx_HE",wx_H)
        return 0, wx_H,

    '''
    grpc，server端，将wx返回'''

    def remote_wx(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "remote_wx":
            # print(reqdata)
            self.uuid = uuid
            wx = self.generate_wx()
            # print("hhhhhhh=",wx)
            # print(wx)
            return 0, wx
        else:
            LOGGER.info("remote_wx Error!")
            return 500, "remote_wx Error!"

    '''
        grpc，server端，接受加密的残差residual，返回加密的梯度div_j_H'''

    def get_residual(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "get_residual":
            # print("hhhhhhhhhhh=",reqdata)
            self.uuid = uuid
            residual = reqdata  # 拿到加密d
            m = len(residual)
            residual = np.array(residual).reshape(m, 1)

            # print(np.shape(wx_H))
            x = self.data[:, 1:]  # 明文特征矩阵
            # print(np.shape(x))
            div_j_H = lin_basic.gradient(residual, x)  # 加密的梯度
            div_j_H = list(map(lambda x: x[0], div_j_H.tolist()))
            # print(div_j_H)
            return 0, div_j_H
        else:
            LOGGER.info("get_residual Error!")
            return 500, "get_residual Error!"

    '''
        grpc，server端，接收解密后的梯度div_j_H，返回更新后的wx值'''

    def get_div_j(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "linr_req_div_j":
            self.uuid = uuid
            div_j_H = reqdata[0]  # 拿到加密梯度
            m = len(div_j_H)
            div_j_H = np.array(div_j_H).reshape(m, 1)
            alpha = reqdata[1]
            x = self.data[:, 1:]  # 明文特征矩阵
            self.w = lin_basic.update_w(self.w, div_j_H, alpha)
            # print(np.shape(self.w))
            # print("newWWWW",self.w)
            wx_H = lin_basic.wx(x, self.w)
            wx_H = list(map(lambda x: x[0], wx_H.tolist()))

            return 0, wx_H
            # print("hhhhhh",wx_H)
        else:
            LOGGER.info("linr_req_div_j Error!")
            return 500, "linr_req_div_j Error!"


if __name__ == '__main__':
    lh = lin_host()
    wx = lh.generate_wx(file)
    # print(wx)
