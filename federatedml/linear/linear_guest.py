import numpy as np

from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from federatedml.linear.linear_basic import *
from federatedml.linear.linear_basic import converged
import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi
import multiprocessing as mp
import csv
import pandas as pd


class linear_guest(linear_basic, PaillierEncrypt):
    def __init__(self):
        super().__init__()
        # self.uuid = None
        self.public_key = None
        self.privacy_key = None
        self.stub = None
        self.grpcclient = None
        self.data = None
        self.alldata = None
        self.testdata = None
        self.w = None

    '''
    获取数据，求交后的数据
    数据格式为
    '''

    def loaddata(self, file):
        df = pd.read_csv(file)
        pp = df.values
        #
        pp = np.hstack((df, np.ones((np.shape(df)[0], 1))))  # 给x添加x[0]项
        self.alldata = pp
        self.data = self.alldata
        # return self.data

    # 第一步：生成pk
    def linr_GeneratePK(self):
        encrypt_operator = PaillierEncrypt()
        encrypt_operator.generate_key()
        self.public_key = encrypt_operator.get_public_key()
        self.privacy_key = encrypt_operator.get_privacy_key()

    def shuffle_data(self, sample):
        m, n = np.shape(sample)
        indexs = random.sample(range(m), m)
        return indexs

    def model_save(self, name, path, parameter):
        np.save(path + name + "_guest", parameter)

    def model_test(self, name, path, dataname='data', datapath=r'train_test_data/'):
        w = np.load(path + name + "_guest.npy")
        data = np.load(datapath + dataname + '_guest.npz')
        xg = data['arr_1'][:, 2:]
        m, n = np.shape(xg)
        print(w)
        wx_G = np.matmul(xg, w)
        reqdata = [name, path, dataname, datapath]
        req = self.grpcclient.request_from_OnetoOne(trancode='test_wx_H', uuid="uuid", reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # 拿到初始化wx_H
        y = wx_H + wx_G
        y_test = data['arr_1'][:, 1].reshape(m, 1)  ### 求了预测的y，其他的诸如
        return {'mse': mean_squared_error(y_test, y), 'rmse': root_mean_squared_error(y_test, y),
                'mae': mean_absolute_error(y_test, y),
                'r2': r_squared(y_test, y)}  #### 这里返回一个字典给张鸣皓

    # 训练模型
    def fit(self, epoch=1000, alpha=0.01, batchSize=100, train_part=0.3, model_name="testmodel", model_path=r'log/'
            , dataname='data', datapath=r'train_test_data/', threshold=0.00001, regular=0, attenuation=1):
        self.grpcclient = modelClientApi.GrpcClient("appconf.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        self.linr_GeneratePK()

        ''' shuffle data '''
        indexs = self.shuffle_data(self.alldata)

        self.alldata = self.alldata[indexs]

        req = self.grpcclient.request_from_OnetoOne(trancode='shuffle_data', uuid='uuid_shuffle_data',
                                                    reqdata=indexs)
        response = self.stub.OnetoOne(req)

        '''   切分训练集与测试集  '''
        self.data, self.testdata = linear_basic.train_test(self.alldata, train_part)
        np.savez(datapath + dataname + '_guest.npz', self.data, self.testdata)
        reqdata = [datapath + dataname, train_part]
        req = self.grpcclient.request_from_OnetoOne(trancode='train_test_div', uuid='uuid_train_test_div',
                                                    reqdata=reqdata)
        response = self.stub.OnetoOne(req)

        useall, indexs = linear_basic.get_random_sample(self.data, batchSize)

        batch_data = linear_basic.get_index_array(self.data, useall, indexs)
        if useall == 1:
            req = self.grpcclient.request_from_OnetoOne(trancode='linear_random_index_array', uuid='uuid_for_index',
                                                        reqdata=indexs)
            response = self.stub.OnetoOne(req)
        m = np.shape(batch_data)[0]
        y = batch_data[:, 1].reshape(m, 1)
        x = batch_data[:, 2:]

        init_w_G = linear_basic.init_w(x)
        # guest_w = init_w_G
        self.w = init_w_G
        wx_G = np.matmul(x, init_w_G)  # 获取初始化wx_G

        uuid = file.split('\\')[-1]  # win\\，linux\
        req = self.grpcclient.request_from_OnetoOne(trancode='linear_remote_wx', uuid=uuid, reqdata='reqdata')
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # 拿到初始化wx_H

        m = len(wx_H)
        wx_H = np.array(wx_H).reshape(m, 1)  # 转换成矩阵

        converged.set_threshold(threshold)  # 设置当模型变化太小的时候，就当作模型已经收敛

        # 梯度下降
        for i in range(epoch):
            alpha = alpha * attenuation  ### 每次循环都微调alpha的数值

            residual = linear_basic.compute_d(wx_G, wx_H, y)  # 计算残差

            lst_residual = list(map(lambda x: x[0], residual.tolist()))
            loss = linear_basic.jtheta(residual)  # 计算loss

            converged.mean_10(loss=loss)
            '''当模型收敛的时候，需要将模型保存  需要输入模型名称、模型保存路径'''
            if converged.stop_or_ahead():  # 判断收敛
                reqdata = [model_name, model_path]
                req = self.grpcclient.request_from_OnetoOne(trancode='model_save', uuid='uuid_model_save',
                                                            reqdata=reqdata)
                response = self.stub.OnetoOne(req)
                self.model_save(model_name, model_path, parameter=self.w)
                break

            div_j = linear_basic.gradient(residual, x)  # 计算梯度
            self.w = linear_basic.update_w(self.w, div_j, alpha)  # 更新w_G
            print(self.w)

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
            req = self.grpcclient.request_from_OnetoOne(trancode='linear_get_residual', uuid=uuid,
                                                        reqdata=residual)  # 发送残差residual
            response = self.stub.OnetoOne(req)
            div_j_H = pickle.loads(response.respdata)  # 拿到host端加密梯度

            '''第1.5步：更新batch_data'''
            useall, indexs = linear_basic.get_random_sample(self.data, batchSize)
            batch_data = linear_basic.get_index_array(self.data, useall, indexs)

            if useall == 1:
                req = self.grpcclient.request_from_OnetoOne(trancode='linear_random_index_array', uuid='uuid_for_index',
                                                            reqdata=indexs)
                response = self.stub.OnetoOne(req)

            m = np.shape(batch_data)[0]
            y = batch_data[:, 1].reshape(m, 1)
            x = batch_data[:, 2:]
            wx_G = linear_basic.wx(x, self.w)

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
            req = self.grpcclient.request_from_OnetoOne(trancode='linear_req_div_j', uuid=uuid, reqdata=reqdata)
            response = self.stub.OnetoOne(req)
            wx_H = pickle.loads(response.respdata)  # 拿到更新后的wx_H
            m = len(wx_H)
            wx_H = np.array(wx_H).reshape(m, 1)  # 转换成矩阵
        return self.model_test(name=model_name, path=model_path, dataname=dataname,
                               datapath=datapath), model_path + model_name, datapath + dataname
    ###  返回模型评价指标,模型路径，数据保存路径（test与train放在一个npz文件里。）

    # 保存模型
    # def save_model(self,):


if __name__ == '__main__':
    # security_level, block_num, random_bit = readbasesecureinfo("appconf.yaml")
    file = "F:\\datasource\\linr_data_y.csv"
    lg = linear_guest()
    lg.loaddata(file)
    lg.fit(1000, 0.0001, -1, 1, "model1", r'log/', threshold=0.001)
