import argparse
import ast

from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.extprint import ExPrint
from federatedml.woe.woe_iv_ca_square_basic import *
from baseCommon.pymysqlclass import *
import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi
import multiprocessing as mp
import csv
import pandas as pd


class woe_iv_ca_square_guest(PaillierEncrypt):
    def __init__(self):
        super().__init__()
        # self.uuid = None
        self.public_key = None
        self.privacy_key = None
        self.stub = None
        self.grpcclient = None
        self.data = None
        # 20210623
        self.jobid = None
        self.name = None
        self.start_time = None
        self.guestcsv = None
        self.hostcsv = None
        self.guestcolum = None
        self.hostcolum = None
        self.getoutfile = None
        self.hostoutfile = None
        self.boxCount = None
        self.boxMethod = None

        self.method = 'mean_num_descrete'

    # 第一步：生成pk
    def linr_GeneratePK(self):
        encrypt_operator = PaillierEncrypt()
        encrypt_operator.generate_key()
        self.public_key = encrypt_operator.get_public_key()
        self.privacy_key = encrypt_operator.get_privacy_key()

    def assign_init_woe_iv_para(self, jobid, name, start_time, guestfile, hostfile, guestcolum, hostcolum, getoutfile,
                                hostoutfile, boxCount, boxMethod,project_id):
        self.jobid = jobid  #
        self.name = name  #
        self.start_time = start_time
        self.guestcsv = guestfile
        self.hostcsv = hostfile  #
        self.guestcolum = guestcolum
        self.hostcolum = hostcolum
        self.getoutfile = getoutfile
        self.hostoutfile = hostoutfile  #
        self.boxCount = boxCount
        self.boxMethod = boxMethod
        self.orGuestDataName = inter_dataname_TANS(guestfile)
        # self.orfilePath = get_path_by_name(self.orGuestDataName)
        self.orfilePath = get_path_from_db(self.orGuestDataName)
        self.guestcsv = concat_filepath(guestfile, jobid, self.orfilePath)
        self.project_id=project_id

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        encrypt_y = list(map(lambda x: self.public_key.encrypt(x), self.data.values[:, 1]))
        # encrypt_y = self.data.values[:, 1]
        reqdatatmp = (encrypt_y, self.jobid, self.name, self.hostcsv, self.hostoutfile)
        ExPrint.extdebug("load_data:" + str(reqdatatmp[1:]))
        req = self.grpcclient.request_from_OnetoOne(trancode='load_data', uuid='uuid', reqdata=reqdatatmp)
        response = self.stub.OnetoOne(req)

    '''
    离散型变量在host端的处理
    1，离散型变量的编码
    2，编码后离散型数据的卡方分箱
    3，将分箱后数据执行woe转换
    '''

    def discrete_feacher(self, feacher, category, ca_threshold, bin_num_threshold):
        reqdata = [feacher]
        req = self.grpcclient.request_from_OnetoOne(trancode='discrete_feacher', uuid='uuid', reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        feacher_category_index, y_list_encrypt, feacher_category = pickle.loads(response.respdata)
        y_list = []
        for list1 in y_list_encrypt:
            list1 = list(map(lambda x: self.privacy_key.decrypt(x), list1))
            y_list.append(list1)

        combination_list = []
        while True:
            combination, min_ca_square = get_mini_discrete_combination(category, y_list, feacher_category_index)
            if min_ca_square > ca_threshold and len(y_list) <= bin_num_threshold:
                break
            combination_list.append(combination)
            y_list, feacher_category_index = feacher_and_y_list_extend(y_list, feacher_category_index, combination)
        dictionary = combination_dictionary(combination_list, feacher_category, feacher_category_index)
        reqdata = [feacher, dictionary, feacher_category]
        print(dictionary)
        req = self.grpcclient.request_from_OnetoOne(trancode='discrete_combination', uuid='uuid', reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        return [feacher_category, dictionary]

    '''
    离散型变量在guest端的处理
    1，离散型变量的编码
    2，编码后离散型数据的卡方分箱
    3，将分箱后数据执行woe转换
    '''

    def guest_discrete_feacher(self, feacher, category, ca_threshold, bin_num_threshold):
        self.data, feacher_category = discrete_encode(self.data, feacher)
        feacher_category_index = list(set(self.data[feacher]))
        y_list = discrete_ylist(self.data, feacher, feacher_category_index)
        combination_list = []
        while True:
            combination, min_ca_square = get_mini_discrete_combination(category, y_list, feacher_category_index)
            if min_ca_square > ca_threshold and len(y_list) <= bin_num_threshold:
                break
            combination_list.append(combination)
            y_list, feacher_category_index = feacher_and_y_list_extend(y_list, feacher_category_index, combination)
        dictionary = combination_dictionary(combination_list, feacher_category, feacher_category_index)
        m, n = self.data.shape
        for i in range(m):
            for comb in dictionary:
                a = int(comb)
                b = int(self.data.loc[i, feacher])
                if feacher_category[b] in dictionary[a]:
                    self.data.loc[i, feacher] = comb
        return [feacher_category, dictionary]

    '''
    连续型变量在host端的处理
    1，连续型变量等频分箱
    2，等频分箱后进行的卡方分箱
    3，将分箱后数据执行woe转换
    '''

    def continuous_feacher(self, feacher, category, ca_threshold, bin_num_threshold):
        reqdata = self.method, feacher, bin_num_threshold
        req = self.grpcclient.request_from_OnetoOne(trancode='continuous_feacher', uuid='uuid', reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        feacher_list, y_list_encrypt = pickle.loads(response.respdata)
        y_list = []
        for list1 in y_list_encrypt:
            list1 = list(map(lambda x: self.privacy_key.decrypt(x), list1))
            y_list.append(list1)
        print('这个是解密后的list', y_list)

        cutlist = rangewo(len(y_list))
        while True:
            flag, cutlist, y_list = del_min_ca_square(y_list, cutlist, ca_threshold, bin_num_threshold, category)
            if flag == False:
                break
        feacher_cut_point_list = []
        for i in cutlist:
            feacher_cut_point_list.append(feacher_list[i])
        reqdata = feacher, feacher_cut_point_list
        req = self.grpcclient.request_from_OnetoOne(trancode='continue_combination', uuid='uuid', reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        return feacher_cut_point_list

    '''
    连续型变量在guest端的处理
    1，连续型变量等频分箱
    2，等频分箱后进行的卡方分箱
    3，将分箱后数据执行woe转换
    '''

    def guest_continuous_feacher(self, feacher, category, ca_threshold, bin_num_threshold):
        feacher_list, y_list = give_bin_ypoint_and_ylist(self.method, self.data, feacher, bin_num_threshold * 20)
        cutlist = rangewo(len(y_list))
        while True:
            flag, cutlist, y_list = del_min_ca_square(y_list, cutlist, ca_threshold, bin_num_threshold, category)
            if flag == False:
                break
        feacher_cut_point_list = []
        for i in cutlist:
            feacher_cut_point_list.append(feacher_list[i])
        self.data = continuation_combination(feacher, self.data, feacher_cut_point_list)
        return feacher_cut_point_list

    '''
    guest端的woe转换
    计算对应某一箱的woe的数值
    计算该特征的iv数值
    '''

    def guest_woe_transform(self, feacher):
        feacher_category = list(set(self.data[feacher]))
        y_list = discrete_ylist(self.data, feacher, feacher_category)
        woelist, iv = calculate_feacher_woeiv(self.data, y_list)
        self.data = woe_transform(feacher_category, woelist, self.data, feacher)
        print(self.data, '这是guest %s 的最终结果' % feacher)
        # for i in range(len(feacher_category)):
        #     feacher_category[i]=feacher_category[i]+1
        return [feacher_category, woelist, iv]

    '''
    host端的woe转换
    计算对应某一箱的woe的数值
    计算该特征的iv数值
    '''

    def host_woe_transform(self, feacher):
        reqdata = feacher
        req = self.grpcclient.request_from_OnetoOne(trancode='host_woe_transform_get_y_list', uuid='uuid',
                                                    reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        y_list_encrypt, feacher_category = pickle.loads(response.respdata)
        y_list = []
        for list1 in y_list_encrypt:
            list1 = list(map(lambda x: self.privacy_key.decrypt(x), list1))
            y_list.append(list1)
        woelist, iv = calculate_feacher_woeiv(self.data, y_list)
        reqdata = feacher, woelist, iv, feacher_category
        req = self.grpcclient.request_from_OnetoOne(trancode='host_woe_transform_give_woe', uuid='uuid',
                                                    reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        # for i in range(len(feacher_category)):
        #     feacher_category[i]=feacher_category[i]+1
        return [feacher_category, woelist, iv]

    '''
    组合调用host端的功能组完成所有host端特征的分箱与woe转换
    '''

    def fit_host(self, feacher_list, category, ca_threshold, bin_num_threshold):
        save_woe_translate_host = {}
        for feacher in feacher_list:
            if feacher[1] == '1':
                hcf = self.continuous_feacher(feacher[0], category, ca_threshold, bin_num_threshold)
                save_woe_translate_host[feacher[0]] = [1, hcf]
                ExPrint.extdebug("特征 " + str(feacher) + "完成了分箱操作")
            else:
                hdf = self.discrete_feacher(feacher[0], category, ca_threshold, bin_num_threshold)
                save_woe_translate_host[feacher[0]] = [2, hdf]
                ExPrint.extdebug("特征 " + str(feacher) + "完成了分箱操作")
            hwt = self.host_woe_transform(feacher[0])
            ExPrint.extdebug("特征 " + str(feacher) + "完成了woe转换")
            save_woe_translate_host[feacher[0]].append(hwt)
            ivgo = save_woe_translate_host[feacher[0]][2][2]
            feacher.append(formatstr(ivgo))
        ExPrint.extdebug("host端全部的特征都完成了woe转换")
        return feacher_list, save_woe_translate_host

    '''
    组合调用guest端的功能组完成所有guest端特征的分箱与woe转换
    '''

    def fit_guest(self, feacher_list, category, ca_threshold, bin_num_threshold):
        save_woe_translate_guest = {}
        ExPrint.extdebug("________________________12123123123")
        for feacher in feacher_list:
            ExPrint.extdebug("打印feature" + str(feacher))
            if feacher[1] == '1':
                gcf = self.guest_continuous_feacher(feacher[0], category, ca_threshold, bin_num_threshold)
                save_woe_translate_guest[feacher[0]] = [1, gcf]
                ExPrint.extdebug("特征 " + str(feacher) + "完成了分箱操作")
            else:
                test_feature_len = list(set(self.data[feacher[0]]))
                if len(test_feature_len)>=200:
                    mysql_iv_model_status("failed",self.jobid)
                    break

                gdf = self.guest_discrete_feacher(feacher[0], category, ca_threshold, bin_num_threshold)
                save_woe_translate_guest[feacher[0]] = [2, gdf]
                ExPrint.extdebug("特征 " + str(feacher) + "完成了分箱操作")
            gwt = self.guest_woe_transform(feacher[0])
            ExPrint.extdebug("特征 " + str(feacher) + "完成了woe转换")
            save_woe_translate_guest[feacher[0]].append(gwt)
            ivgo = save_woe_translate_guest[feacher[0]][2][2]
            feacher.append(formatstr(ivgo))
        print(save_woe_translate_guest)
        save_path = self.orfilePath
        filename = self.orGuestDataName + "_inter_woe.csv"
        save_path = concat_filepath(filename, self.jobid, save_path)
        self.data.to_csv(save_path, index=0)
        ExPrint.extdebug("Guest端全部的特征都完成了woe转换")
        return feacher_list, save_woe_translate_guest

    def fit(self, feacher_list_host, feacher_list_guest, category, ca_threshold, bin_num_threshold):
        mysql_iv_model_status("start", self.jobid)
        self.grpcclient = modelClientApi.GrpcClient("woe_iv_ca_Model.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        self.linr_GeneratePK()
        self.load_data(self.guestcsv)
        ExPrint.extdebug("load_data " + self.guestcsv + "finished!!")
        feacher_list_guest, save_woe_translate_guest = self.fit_guest(feacher_list_guest, category, ca_threshold,
                                                                      bin_num_threshold)
        ExPrint.extdebug("fit_guest   finished!!")
        feacher_list_host, save_woe_translate_host = self.fit_host(feacher_list_host, category, ca_threshold,
                                                                   bin_num_threshold)
        ExPrint.extdebug("fit_host   finished!!")
        save_woe_translate = [save_woe_translate_guest, save_woe_translate_host]

        '''
        保存woe的流程控制数据 process_control_data
        '''

        save_controll_data(save_woe_translate, self.jobid, 'woe')

        to_file(get_path() + "%s_detail_woe.txt" % self.jobid, save_woe_translate)
        mysql_iv_result(feacher_list_host, feacher_list_guest, inter_dataname_TANS(self.hostcsv), self.orGuestDataName, self.project_id, self.jobid)
        mysql_iv_model_status("finished", self.jobid)
        return feacher_list_host, feacher_list_guest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="woe_iv_ca_square_guest")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-start_time", "--start_time", type=str, required=True)
    parser.add_argument("-guestcsv", "--guestcsv", type=str, required=True)
    parser.add_argument("-hostcsv", "--hostcsv", type=str, required=True)
    parser.add_argument("-guestcsv_column_name", "--guestcsv_column_name", type=str, required=True)
    parser.add_argument("-hostcsv_column_name", "--hostcsv_column_name", type=str, required=True)
    parser.add_argument("-getoutfile", "--getoutfile", type=str, required=False)
    parser.add_argument("-hostoutfile", "--hostoutfile", type=str, required=True)
    parser.add_argument("-boxCount", "--boxCount", type=str, required=True)
    parser.add_argument("-boxMethod", "--boxMethod", type=str, required=True)
    parser.add_argument("-project_id","--project_id",type=str,required=True)
    args = parser.parse_args()

    jobid = args.id
    name = args.name
    start_time = args.start_time
    guesttable = args.guestcsv
    hosttable = args.hostcsv
    guestcolum = args.guestcsv_column_name
    hostcolum = args.hostcsv_column_name
    getoutfile = args.getoutfile
    hostoutfile = args.hostoutfile
    boxCount = args.boxCount
    boxMethod = args.boxMethod
    project_id = args.project_id

    #
    # datax=pd.read_csv("/data/zhanghui/python/federatedml/linear/log/logistic_data_x.csv")
    # for i in range(10000):
    #     datax.loc[i, 'x6'] = (int(datax.loc[i, 'x4'] * 100))
    #     datax.loc[i, 'x7'] = str(int(datax.loc[i, 'x5'] * 100))
    # datax.to_csv("/data/zhanghui/python/federatedml/linear/log/logistic_data_x_descret.csv", index=0)

    ExPrint.extdebug("woe_iv_mode  Start ...")
    ExPrint.extdebug("get woe_iv_mode json data start ....")
    tst = woe_iv_ca_square_guest()

    feacher_list_host = ast.literal_eval(hostcolum)
    feacher_list_guest = ast.literal_eval(guestcolum)

    ExPrint.extdebug("assign_init_woe_iv_para  start..")
    tst.assign_init_woe_iv_para(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum, getoutfile,
                                hostoutfile, boxCount, boxMethod,project_id)
    myDataIv, coDataIv = tst.fit(feacher_list_host, feacher_list_guest, [0, 1], 0.99, 5)
    ExPrint.saveJsondata("myDataIv=" + str(myDataIv))
    ExPrint.saveJsondata("coDataIv=" + str(coDataIv))
    # print(myDataIv)
