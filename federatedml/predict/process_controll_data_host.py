import numpy as np

from baseCommon.plogger import *
from federatedml.predict.predict_bisic import *
from baseCommon.pymysqlclass import *
# from secureprotol.encrypt import PaillierEncrypt
from baseCommon.logger import LogClass, ON
import pandas as pd
import multiprocessing as mp

LoggerFactory.set_directory(directory="./")
LOGGER = getLogger()


class process_controll_data_host():
    def __init__(self):
        self.jobid = None
        self.host_filename = None
        self.data = None
        self.filepath = None

    def test(self):
        print('hello grpc')

    def get_controll_data(self, jobid, method):
        save_path = getpath_by_jobid(jobid, method)
        controll_data = pickle_load(save_path)
        return controll_data

    def test_grpc(self, tradecode, uuid, reqdata):
        if tradecode == "test_grpc":
            print('hello_grpc')
            return 0, 'test_grpc ok go'
        else:
            return 500, "test_grpc Error!"

    ''' host端完成数据载入'''

    def get_path_from_db(self,filename):
        sqlc = mysqlClass()
        param = [filename]
        sql = "select file_path from smpc_data_info where data_name = %s"
        filepath = sqlc._fetchone(sql, param=param)[0]
        return filepath


    def pload_data(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "pload_data":
            (encrypt_y, self.jobid, self.host_filename,predict_id) = reqdata
            # self.filepath = get_file_path_by_http( self.host_filename)
            orHostDataName = inter_dataname_TANS(self.host_filename)
            # filePath = get_path_by_name(orHostDataName)
            filePath = self.get_path_from_db(orHostDataName)
            self.filepath = concat_filepath(self.host_filename, predict_id, filePath)
            self.data = pd.read_csv(self.filepath)
            self.data = insert_encrypt_y(self.data, encrypt_y)
            return 0, 'succes'
        else:
            LOGGER.info("pload_data Error!")
            return 500, "pload_data Error!"

    ''' 利用guest端传回的离散型数据编码方法，箱合并信息，操作host端的数据完成分箱编码操作 '''

    def pdiscrete_combination(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "pdiscrete_combination":
            feacher, dictionary, feacher_category = reqdata
            m, n = self.data.shape
            for i in range(m):
                for comb in dictionary:
                    chenge_to_int=int(self.data.loc[i, feacher])
                    if feacher_category[chenge_to_int] in dictionary[comb]:
                        self.data.loc[i, feacher] = comb
            # print(self.data,set(self.data.loc[:,feacher]))
            # self.save_woe_translate[feacher] = [2, [feacher_category,dictionary]]
            return 0, 'success'
        else:
            LOGGER.info("pdiscrete_combination Error!")
            return 500, "pdiscrete_combination Error!"

    ''' 利用guest端传回的连续型数据分箱信息，操作host端的数据完成分箱编码操作 '''

    def pcontinue_combination(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "pcontinue_combination":
            feacher, feacher_cut_point_list = reqdata
            # self.save_woe_translate[feacher]=[1,feacher_cut_point_list]
            self.data = continuation_combination(feacher, self.data, feacher_cut_point_list)
            print(self.data, set(self.data.loc[:, feacher]))
            return 0, 'success'
        else:
            LOGGER.info("pcontinue_combination Error!")
            return 500, "pcontinue_combination Error!"

    ''' 利用guest端传回的woe转换信息，操作host端相应特征的数据完成woe转换'''

    def phost_woe_transform_give_woe(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "phost_woe_transform_give_woe":
            feacher, woelist, iv, feacher_category = reqdata
            self.data = woe_transform(feacher_category, woelist, self.data, feacher)
            # self.save_woe_translate[feacher].append([feacher_category, woelist,iv])
            # print(self.save_woe_translate)
            # print(self.data, '这是host %s 的最终结果'%feacher)
            return 0, iv
        else:
            LOGGER.info("phost_woe_transform_give_woe Error!")
            return 500, "phost_woe_transform_give_woe Error!"

    ''' 调用保存在host端的w，计算wx_G并返回给guest端'''

    def p_wx_H(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "p_wx_H":
            self.train_controll = self.get_controll_data(self.jobid, 'train_host')
            w = self.train_controll['para']
            columns = self.train_controll["column"]
            columns = st2list(columns)

            self.data = self.data[columns]
            dfarr = np.array(self.data)[:, 1:]
            pp = np.hstack((dfarr, np.ones((np.shape(dfarr)[0], 1))))  # 给x添加x[0]项
            wx_G = np.matmul(pp, w)

            return 0, wx_G
        else:
            LOGGER.info("p_wx_H Error!")
            return 500, "p_wx_H Error!"
