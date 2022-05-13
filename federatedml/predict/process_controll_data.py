import os
import numpy as np
import pandas as pd
import time
from collections import Counter
import random
import pickle
import argparse
import ast
from federatedml.predict.predict_bisic import *
from federatedml.logistic.logistic_basic import *
from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.projectConf import get_project_base_directory
from baseInterface import utilsApi, model_pb2_grpc, modelClientApi
from baseCommon.extprint import ExPrint
from baseCommon.pymysqlclass import *

class process_controll():
    def __init__(self):
        self.predict_id=None
        self.jobid = None
        self.guest_controll_data = None
        self.host_controll_data = None
        self.host_filename = None
        self.guest_filename = None
        self.woe_controll = None
        self.train_controll = None

    def get_path_from_db_guest(self,predict_id):
        sqlc = mysqlClass()
        param = [predict_id]
        sql = "select file_path from smpc_predict where id = %s"
        filepath = sqlc._fetchone(sql, param=param)[0]
        return filepath
    def assign_paramerter(self, jobid, guest_filename, host_filename,predict_id):
        self.host_filename = host_filename
        self.jobid = jobid
        self.guest_filename = guest_filename
        self.predict_id=predict_id
        orGuestDataName = inter_dataname_TANS(guest_filename)
        # self.orfilePath = get_path_by_name(orGuestDataName)
        self.orfilePath = self.get_path_from_db_guest(self.predict_id)
        self.guestcsv = concat_filepath(guest_filename, self.predict_id, self.orfilePath)
        self.result_path=concat_filepath(orGuestDataName+'predict_result.csv', jobid, self.orfilePath)

    '''
    供训练过程调用的保存中间流程控制数据的函数，以jobid与method确定文件全路径。
    '''

    def save_controll_data(controll_data, jobid, method):
        save_path = getpath_by_jobid(jobid, method)
        pickle_save(save_path, controll_data)

    '''
    以jobid与method确定文件全路径，并读取保存的中间流程控制数据
    '''

    def get_controll_data(self, jobid, method):
        save_path = getpath_by_jobid(jobid, method)
        controll_data = pickle_load(save_path)
        return controll_data

    def grpc_test(self, jobid):
        print('这里就是jobid' + str(jobid))
        self.grpcclient = modelClientApi.GrpcClient("pridict.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        req = self.grpcclient.request_from_OnetoOne(trancode='test_grpc', uuid='uuid', reqdata='test_grpc')
        response = self.stub.OnetoOne(req)
        print("rsp=", response)
        returntext = pickle.loads(response.respdata)
        print(' 完成咯', returntext)
        return 'yes'

    """
    导入要预测的样本文件，并对其做预处理：在第二列插入一列y值（无意义的），使得数据结构与训练时样本数据结构一致。
    """

    def load_data(self):
        # filepath=get_file_path_by_http(self.guest_filename)  ##利用http，通过数据名称获取数据路径
        filepath = self.guestcsv
        self.data = pd.read_csv(filepath)

        self.data = insert_mask_y(self.data, self.data.iloc[:, 0])
        # encrypt_y = list(map(lambda x: self.public_key.encrypt(x), self.data.values[:, 1]))
        encrypt_y = self.data.values[:, 1]
        reqdatatmp = (encrypt_y, self.jobid, self.host_filename,self.predict_id)  # guest将参数传往host端
        req = self.grpcclient.request_from_OnetoOne(trancode='pload_data', uuid='uuid',
                                                    reqdata=reqdatatmp)  # guest将参数传往host端
        response = self.stub.OnetoOne(req)  # guest将参数传往host端
        ##host导入数据
        self.columns_ori = self.data.columns

    '''host 部分 判断特征连续型或离散型，并做相应分箱处理，最后根据分箱结果完成woe转换'''

    def host_bin_woe_translate(self, save_woe_translate_host):
        for feacher in save_woe_translate_host:
            if save_woe_translate_host[feacher][0] == 1:  # 特征为连续型
                hcf = save_woe_translate_host[feacher][1]
                feacher_cut_point_list = hcf
                reqdata = feacher, feacher_cut_point_list
                req = self.grpcclient.request_from_OnetoOne(trancode='pcontinue_combination', uuid='uuid',
                                                            reqdata=reqdata)
                response = self.stub.OnetoOne(req)
                ExPrint.extdebug(feacher + "host_continuous_woe finished!!")
            else:  # 特征为离散型
                hdf = save_woe_translate_host[feacher][1]
                feacher_category, dictionary = hdf
                reqdata = [feacher, dictionary, feacher_category]
                req = self.grpcclient.request_from_OnetoOne(trancode='pdiscrete_combination', uuid='uuid',
                                                            reqdata=reqdata)
                response = self.stub.OnetoOne(req)
                ExPrint.extdebug(feacher + " host_discrete_woe finished!!")
            hwt = save_woe_translate_host[feacher][2]
            feacher_category, woelist, iv = hwt
            reqdata = feacher, woelist, iv, feacher_category
            req = self.grpcclient.request_from_OnetoOne(trancode='phost_woe_transform_give_woe', uuid='uuid',
                                                        reqdata=reqdata)
            response = self.stub.OnetoOne(req)
            print('现在完成了host端feature：', feacher, '的woe转换')
            ExPrint.extdebug(" host_woe_transform finished!!")

    '''guest 部分调用函数   判断特征连续型或离散型，并做相应分箱处理，最后根据分箱结果完成woe转换'''

    def guest_bin_woe_translate(self, save_woe_translate_guest):
        for feacher in save_woe_translate_guest:
            if save_woe_translate_guest[feacher][0] == 1:  # 特征为连续型

                gcf = save_woe_translate_guest[feacher][1]
                feacher_cut_point_list = gcf
                print(feacher, feacher_cut_point_list)
                self.data = continuation_combination(feacher, self.data, feacher_cut_point_list)
                ExPrint.extdebug(feacher + "guest_continuous_woe finished!!")
            else:  # 特征为离散型
                gdf = save_woe_translate_guest[feacher][1]
                feacher_category, dictionary = gdf
                m, n = self.data.shape
                for i in range(m):
                    for comb in dictionary:
                        selfdate_toint=int(self.data.loc[i, feacher])
                        if feacher_category[selfdate_toint] in dictionary[comb]:
                            self.data.loc[i, feacher] = comb
            gwt = save_woe_translate_guest[feacher][2]
            feacher_category, woelist, iv = gwt
            print(feacher_category, woelist, self.data)
            self.data = woe_transform(feacher_category, woelist, self.data, feacher)  # 完成woe转换
            print('现在完成了guest端feature：', feacher, '的woe转换')
            ExPrint.extdebug(feacher + "guest_discrete_woe finished!!")

    # def ifContinuous(data):
    '''回归部分   取出对应类型的w，生成g端和h端的wx，根据不同的回归算法得到对应的y'''

    def regression(self):
        self.train_controll = self.get_controll_data(self.jobid, 'train_guest')
        w = self.train_controll['para']
        method = self.train_controll['method']
        columns = self.train_controll["column"]
        columns= st2list(columns)
        columns_data=list(self.columns_ori)
        columnm1=[columns_data[0],columns_data[1]]
        columnm1=columnm1+columns[2:]
        self.data = self.data[columnm1]
        xg = np.array(self.data)[:, 2:]
        print(self.data, xg, w)
        m, n = np.shape(xg)
        wx_G = np.matmul(xg, w)  # 自己计算wx_G
        reqdata = tst.host_filename + self.jobid
        req = self.grpcclient.request_from_OnetoOne(trancode='p_wx_H', uuid="uuid", reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # 拿到wx_H
        if method == 'linear':
            y_pre = wx_H + wx_G
        elif method == 'logistic':
            y_pre = logistic_basic.sigmoid(wx_H + wx_G)
        ExPrint.extdebug(" predict y finished!!")
        y_pre_binary = np.array(y_pre >= 0.5, dtype=np.int32)
        m, n = y_pre_binary.shape
        y_pre_binary = y_pre_binary.reshape(1, m)[0]
        y_pre = y_pre.reshape(1, m)[0]
        return method, y_pre, y_pre_binary

    def fit(self):
        self.grpcclient = modelClientApi.GrpcClient("pridict.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立grpc通道
        ExPrint.extdebug("grpc established!!")
        sql = 'update smpc_predict set inter_sect_status=%s where id=%s'
        sqlc = mysqlClass()
        sqlc._execute(sql,param=['07',self.predict_id])
        try:
            self.load_data()
            self.woe_controll = self.get_controll_data(self.jobid, 'woe')  # 读取woe相关数据
            save_woe_translate_guest, save_woe_translate_host = self.woe_controll

            '''调用guest部分的处理函数，该函数无返回值，直接对数据进行完整处理'''
            self.guest_bin_woe_translate(save_woe_translate_guest)

            '''调用host部分的处理函数，该函数无返回值，直接对数据进行完整处理'''
            self.host_bin_woe_translate(save_woe_translate_host)

            '''调用回归部分的处理函数，函数返回回归类型于y_pre(y的预测值)'''
            method, y_pre,y_pre_binary = self.regression()
            id_list = list(self.data['id'])
            result_data={'id':id_list,'y_pre':y_pre,'y_pre_binary':y_pre_binary}
            result_data=pd.DataFrame(result_data)
            result_data.to_csv(self.result_path)
        except BaseException:
            sql = 'update smpc_predict set inter_sect_status=%s,predict_result=%s where id=%s'
            sqlc = mysqlClass()
            sqlc._execute(sql, param=['96', self.result_path, self.predict_id])
        sql = 'update smpc_predict set inter_sect_status=%s,predict_result=%s where id=%s'
        sqlc = mysqlClass()
        sqlc._execute(sql,param=['08',self.result_path,self.predict_id])
        print(y_pre,y_pre_binary)
        return self.result_path,self.predict_id


if __name__ == '__main__':
    print("HHHHHHHHHHHHHHH")
    parser = argparse.ArgumentParser(description="woe_iv_ca_square_guest")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-guest_filename", "--guest_filename", type=str, required=True)
    parser.add_argument("-host_filename", "--host_filename", type=str, required=True)
    parser.add_argument("-predict_id","--predict_id",type=str,required=True)
    #
    args = parser.parse_args()
    #
    jobid = args.id
    guest_filename=args.guest_filename
    host_filename=args.host_filename
    predict_id=args.predict_id
    # jobid = "1405343244878561282"
    # guest_filename = "/data/zhanghui/woe/y_test.csv"
    # host_filename = "/data/zhanghui/woe/x.csv"

    tst = process_controll()
    tst.guest_filename = guest_filename
    tst.host_filename = host_filename
    tst.jobid = jobid
    tst.assign_paramerter(jobid,guest_filename,host_filename,predict_id)
    y_pre,id_list=tst.fit()
    tst.assign_paramerter(jobid,guest_filename,host_filename,predict_id)
    result_path,predict_id=tst.fit()
