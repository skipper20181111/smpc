import numpy as np
from federatedml.linr.lin_basic import lin_basic
from federatedml.linr.lin_basic import converged
import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi
import multiprocessing as mp
import csv
import pandas as pd


class lin_guest(lin_basic, PaillierEncrypt):
    def __init__(self):
        super().__init__()
        # self.uuid = None
        self.public_key = None
        self.privacy_key = None
        self.stub = None
        self.grpcclient = None
        self.data = None
        self.w = None

    '''
    获取数据，求交后的数据
    数据格式为
    '''

    def loaddata(self, file):
        df = pd.read_csv(file)
        data = df.values
        self.data = data
        # return self.data

    # 第一步：生成pk
    def linr_GeneratePK(self):
        encrypt_operator = PaillierEncrypt()
        encrypt_operator.generate_key()
        self.public_key = encrypt_operator.get_public_key()
        self.privacy_key = encrypt_operator.get_privacy_key()

    # 训练模型
    def fit(self, epoch, alpha):
        self.grpcclient = modelClientApi.GrpcClient()  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        self.linr_GeneratePK()
        m = np.shape(self.data)[0]
        y = self.data[:, 1].reshape(m, 1)
        x = self.data[:, 2:]

        init_w_G = lin_basic.init_w(x)
        # guest_w = init_w_G
        self.w = init_w_G
        wx_G = np.matmul(x, init_w_G)  # 获取初始化wx_G

        uuid = file.split('\\')[-1]  # win\\，linux\
        req = self.grpcclient.request_from_OnetoOne(trancode='remote_wx', uuid=uuid, reqdata='req_wx')
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # 拿到初始化wx_H
        m = len(wx_H)
        wx_H = np.array(wx_H).reshape(m, 1)  # 转换成矩阵

        # 梯度下降
        for i in range(epoch):
            residual = lin_basic.compute_d(wx_G, wx_H, y)  # 计算残差
            lst_residual = list(map(lambda x: x[0], residual.tolist()))
            loss = lin_basic.jtheta(residual)  # 计算loss
            # conv = converged()
            converged.mean_10(loss=loss)
            # print(converged.stop_or_ahead())
            if converged.stop_or_ahead():  # 判断收敛
                # print(converged.stop_or_ahead())
                break
            div_j = lin_basic.gradient(residual, x)  # 计算梯度
            self.w = lin_basic.update_w(self.w, div_j, alpha)  # 更新w_G
            print(self.w)
            wx_G = lin_basic.wx(x, self.w)
            # print(np.shape(wx_G))

            '''
            两步GRPC传递
            第一步：传递加密残差residualE，拿到加密梯度div_j_H
            '''

            # residualE = list(map(lambda x: self.public_key.encrypt(x), lst_residual))#加密残差
            # with mp.Pool() as pool:
            #     residualE = pool.map(self.public_key.encrypt,lst_residual)
            # print("jiamijiami",residualE)

            uname = 'linr_residual:'
            unum = str(i)
            uuid = uname + unum
            req = self.grpcclient.request_from_OnetoOne(trancode='get_residual', uuid=uuid,
                                                        reqdata=residual)  # 发送残差residual
            response = self.stub.OnetoOne(req)
            div_j_H = pickle.loads(response.respdata)  # 拿到host端加密梯度
            print(1)

            '''第二步：对加密的梯度解密，发送到host端，并拿回新的wx_H'''
            # with mp.Pool() as pool:
            #     div_j_H = pool.map(self.privacy_key.decrypt, div_j_H)

            # div_j_H = list(map(lambda x: self.privacy_key.decrypt(x), div_j_H))

            reqdata = []
            reqdata.append(div_j_H)
            reqdata.append(alpha)
            uname = 'linr_div_j:'
            unum = str(i)
            uuid = uname + unum
            req = self.grpcclient.request_from_OnetoOne(trancode='linr_req_div_j', uuid=uuid, reqdata=reqdata)
            response = self.stub.OnetoOne(req)
            wx_H = pickle.loads(response.respdata)  # 拿到更新后的wx_H
            m = len(wx_H)
            wx_H = np.array(wx_H).reshape(m, 1)  # 转换成矩阵
            print(2)

    # 保存模型
    # def save_model(self,):


if __name__ == '__main__':
    file = "F:\\datasource\\linr_data_y.csv"
    lg = lin_guest()
    lg.loaddata(file)
    lg.fit(1000, 0.0001)
