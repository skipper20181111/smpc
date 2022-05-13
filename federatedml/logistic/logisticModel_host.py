import numpy as np

from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.dataIO import loadcsvFileModedata
from baseCommon.plogger import *
from federatedml.logistic.logistic_basic import *

LoggerFactory.set_directory(directory="./")
LOGGER = getLogger("linear.log")


# interdata = [1,2]#根据求交结果导入
# data = interdata
def getlinearModePathPara(para=None):
    if para is None:
        loadyaml = load_yaml_conf("gobal_conf.yaml")
    else:
        loadyaml = load_yaml_conf(para)
    return getDirectValue(loadyaml, 'host_modelpath'), getDirectValue(loadyaml, 'host_datapath')


class logistic_host(logistic_basic):
    def __init__(self):
        super(logistic_host, self).__init__()
        self.uuid = None
        self.alldata = None
        self.data = None
        self.traindata = None
        self.testdata = None
        self.w = None
        self.testdataname = None
        self.model_name = None
        self.dataname = None
        self.div_h_mean = None
        self.regular = None
        # 2021 add for hosttable
        self.jobid = None
        self.name = None
        self.start_time = None
        self.hosttable = None
        self.hostcolum = None
        self.hostcsv = None

    # 建立配置目录
    def checkpathexist(self):
        model_path, datapath = getlinearModePathPara("logisticModel.yaml")
        project = os.getenv("PROJECTPATH")
        if project is None:
            project = get_project_base_directory()
        model_path = project + "/{}".format(model_path)
        datapath = project + "/{}".format(datapath)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        return model_path, datapath

    # 生成 modelname dataname
    def getModelandDataName(self):
        model_path, datapath = self.checkpathexist()
        modelname = model_path + self.jobid + "_" + self.start_time + "_host.npy"
        dataname = datapath + self.name + self.jobid + "_host.npz"
        return modelname, dataname

    def loadhostscsvdata(self):
        print("self.hostcsv", self.hostcsv)
        datalist = loadcsvFileModedata(self.hostcsv, resetcolumn_1=self.hostcolum)
        datalist = list(map(lambda x: x + [1], datalist))
        # change numpy type
        self.alldata = np.array(datalist)
        return len(datalist)

    '''
    file = "F:\\datasource\\linr_data_x.csv"

    uuid = None
    alldata = None    # linear_host.public_key = None
    data = None
    traindata = None
    testdata = None
    w = None
    
    def loaddata(file):
        df = pd.read_csv(file)
        pp=df.values
        #
        pp = np.hstack((df, np.ones((np.shape(df)[0], 1))))  # 给x添加x[0]项

        linear_host.data = pp
        linear_host.alldata = pp
        return linear_host.alldata
       '''

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

    # get linearModel_guest information
    def func_hostcsvinfo(self, trancode, uuid, reqdata):
        if trancode == "logistic_modelhostcsvinfo":
            hosttabledict = reqdata
            print("hostcsvdict=", reqdata)
            self.jobid = hosttabledict['jobid']
            self.name = hosttabledict['name']
            self.start_time = hosttabledict['start_time']
            self.hostcsv = hosttabledict['hostcsv']
            '''在这里修改hostcsv的地址'''
            orHostDataName = woe_dataname_TANS(self.hostcsv)
            # filePath = get_path_by_name(orHostDataName)
            filePath = get_path_from_db(orHostDataName)
            self.hostcsv = concat_filepath(self.hostcsv, self.jobid, filePath)
            self.hostcolum = hosttabledict['hostcolum']
            self.regular = hosttabledict['regular']
            # assign model_name, dataname values
            if not os.path.exists(self.hostcsv):
                return 5555, "file" + self.hostcsv + " not exist!"
            self.model_name, self.dataname = self.getModelandDataName()
        return 0, "success"

    '''模型保存啊'''

    def logistic_model_save(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_model_save":
            model_path = reqdata
            np.save(self.model_name, self.w)
            model_save_upgrade({"method": "logistic", "para": self.w, "column": self.hostcolum}, self.jobid,
                               method='train_host')
            return 0, reqdata
        else:
            LOGGER.info("logistic_model_save Error!")
            return 500, "logistic_model_save Error!"

    '''这是模型测试'''

    def logistic_test_wx_H(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_test_wx_H":
            id = reqdata
            w = np.load(self.model_name)
            data = np.load(self.dataname)

            wx_G = np.matmul(data['arr_1'][:, 1:], w)

            return 0, wx_G
        else:
            LOGGER.info("logistic_test_wx_H Error!")
            return 500, "logistic_test_wx_H Error!"

    def logistic_modeluse_wx_H(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_modeluse_wx_H":
            id = reqdata
            w = np.load(self.model_name)
            data = np.load(self.testdataname)

            wx_G = np.matmul(data['arr_1'][:, 1:], w)

            return 0, wx_G
        else:
            LOGGER.info("logistic_modeluse_wx_H Error!")
            return 500, "logistic_modeluse_wx_H Error!"

    def logistic_train_test_div(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_train_test_div":
            trainpart = reqdata
            self.traindata, self.testdata = logistic_basic.train_test(self.alldata, trainpart)
            print(f"dataname={self.dataname} trainpart={trainpart}")
            np.savez(self.dataname, self.traindata, self.testdata)
            self.data = self.traindata
            return 0, reqdata
        else:
            LOGGER.info("logistic_train_test_div Error!")
            return 500, "logistic_train_test_div Error!"

    ## 定义
    def logistic_shuffle_data(self, tradecode, uuid, reqdata):
        print(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_shuffle_data" and len(reqdata) > 0:
            # load datafile (from self.hostcsv) to self.alldata
            loadcount = self.loadhostscsvdata()
            # self.loaddata(linear_host.file)
            self.uuid = uuid
            self.alldata = self.alldata[reqdata]  # reqdata 就是 indexs
            if loadcount > 0:
                return 0, "success!"
            else:
                return 555, "loadhostscsvdata" + self.hostcsv + "failed!"
        else:
            LOGGER.info("logistic_shuffle_data Error!")
            return 500, "logistic_shuffle_data Error!"

    ## 定义
    def logistic_random_index_array(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_random_index_array":
            self.uuid = uuid
            self.data = logistic_basic.get_index_array(self.traindata, 1, reqdata)  # reqdata 就是 indexs
            return 0, reqdata
        else:
            LOGGER.info("logistic_random_index_array Error!")
            return 500, "logistic_random_index_array Error!"

    '''
    生成初始化的wx，并转化成list'''

    def generate_wx(self):
        dd = self.data
        x = dd[:, 1:]
        self.w = logistic_basic.init_w(x)
        wx_H = logistic_basic.wx(x, self.w)
        self.div_h_mean = self.w * 0
        # wx_H = wx_H.tolist()
        wx_H = list(map(lambda x: x[0], wx_H.tolist()))
        return wx_H

    '''
    grpc，server端，将wx返回'''

    def remote_wx(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_remote_wx":

            logistic_basic.uuid = uuid
            wx = self.generate_wx()

            return 0, wx
        else:
            LOGGER.info("logistic_remote_wx Error!")
            return 500, "logistic_remote_wx Error!"

    '''
        grpc，server端，接受加密的残差residual，返回加密的梯度div_j_H'''

    def get_residual(self, tradecode, uuid, reqdata):

        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_get_residual":

            self.uuid = uuid
            residual = reqdata  # 拿到加密d
            m = len(residual)
            residual = np.array(residual).reshape(m, 1)
            x = self.data[:, 1:]  # 明文特征矩阵
            div_j_H = logistic_basic.gradient_regul(x, residual, self.regular, 0.00001, self.w, m)  # 计算梯度
            # div_j_H = logistic_basic.gradient(residual, x)#加密的梯度
            div_j_H = list(map(lambda x: x[0], div_j_H.tolist()))

            return 0, div_j_H
        else:
            LOGGER.info("logistic_get_residual Error!")
            return 500, "logistic_get_residual Error!"

    '''
        grpc，server端，接收解密后的梯度div_j_H，返回更新后的wx值'''

    def get_div_j(self, tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "logistic_req_div_j":
            self.uuid = uuid
            div_j_H = reqdata[0]  # 拿到加密梯度
            m = len(div_j_H)
            div_j_H = np.array(div_j_H).reshape(m, 1)
            # self.div_h_mean = 0.9 * self.div_h_mean + 0.1 * div_j_H
            self.div_h_mean = get_mean(self.div_h_mean, div_j_H, 10)
            alpha = reqdata[1]
            x = self.data[:, 1:]  # 明文特征矩阵
            self.w = logistic_basic.update_w(self.w, self.div_h_mean, alpha)
            print(self.w)
            wx_H = logistic_basic.wx(x, self.w)
            wx_H = list(map(lambda x: x[0], wx_H.tolist()))

            return 0, wx_H

        else:
            LOGGER.info("logistic_req_div_j Error!")
            return 500, "logistic_req_div_j Error!"
