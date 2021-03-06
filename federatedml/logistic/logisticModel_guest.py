import argparse
import os

from multiprocessing import Pool
from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.dataIO import loadcsvFileModedata
from baseCommon.extprint import ExPrint
from baseCommon.projectConf import get_project_base_directory
from baseInterface.model_grpcException import callgrpcException, GrpcFailedExit
from federatedml.logistic.logistic_basic import *
from federatedml.logistic.logistic_basic import converged
import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi


class logistic_guest(logistic_basic, PaillierEncrypt):
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
        # 20210521 add for hosttable
        self.dataname = None
        self.model_name = None
        # 20210519 add for hosttable
        self.jobid = None
        self.name = None
        self.start_time = None
        self.hosttable = None
        self.hostcolum = None
        self.hostcsv = None

        # mode param assgion
        self.epoch = 1000
        self.alpha = 0.01
        self.batchSize = -1
        self.train_part = 0.3
        self.threshold = 0.00001
        self.regular = "no"
        self.attenuation = 1

    def assign_init_csvpara(self, jobid, name, start_time, guestfile, hostfile, guestcolum, hostcolum):
        self.jobid = jobid
        self.name = name
        self.start_time = start_time
        self.guestcsv = guestfile
        self.hostcsv = hostfile
        self.guestcolum = guestcolum
        self.hostcolum = hostcolum
        # self.outfilename = outfilename
        self.model_name = jobid + "_" + start_time + "_guest.npy"
        orGuestDataName = woe_dataname_TANS(guestfile)
        print(orGuestDataName, '#######################', guestfile)
        self.orfilePath = get_path_from_db(orGuestDataName)
        self.guestcsv = concat_filepath(guestfile, jobid, self.orfilePath)

    def assign_model_param_value(self, epoch, alpha, batchSize, train_part, threshold, regular, attenuation):
        self.epoch = int(epoch)
        self.alpha = float(alpha)
        self.batchSize = int(batchSize)
        self.train_part = float(train_part)
        self.threshold = float(threshold)
        self.regular = regular
        self.attenuation = float(attenuation)

    def loadcsvdata(self):
        datalist = loadcsvFileModedata(self.guestcsv, resetcolumn_1=self.guestcolum)
        datalist = list(map(lambda x: x, datalist))
        # change numpy type
        self.alldata = np.array(datalist)
        return len(datalist)

    '''
    ?????????????????????????????????
    ???????????????
    
    def loaddata(self,file):
        df = pd.read_csv(file)
        self.alldata = df.values
        self.data = self.alldata
        # return self.data

    '''

    # ??????????????????pk
    def linr_GeneratePK(self):
        encrypt_operator = PaillierEncrypt()
        encrypt_operator.generate_key()
        self.public_key = encrypt_operator.get_public_key()
        self.privacy_key = encrypt_operator.get_privacy_key()

    def shuffle_data(self, npdata):
        m, n = np.shape(npdata)
        indexs = random.sample(range(m), m)
        return indexs

    def model_test(self):
        w = np.load(self.model_name)
        data = np.load(self.dataname)
        xg = data['arr_1'][:, 2:]
        m, n = np.shape(xg)
        wx_G = np.matmul(xg, w)
        reqdata = self.name + self.jobid
        req = self.grpcclient.request_from_OnetoOne(trancode='logistic_test_wx_H', uuid="uuid", reqdata=reqdata)
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # ???????????????wx_H

        y_test = data['arr_1'][:, 1].reshape(m, 1)
        return model_evaluate(wx_H,wx_G,y_test)  #???????????????????????????wx_H,wx_G???????????????????????????y_test???

    def fit(self):
        mysql_iv_model_status('start', self.jobid)
        ExPrint.extdebug('?????????????????????,??????????????????')
        # creat build mode and  data save path
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
        # ??????????????????????????????
        #  assgin build  mode param
        alpha = self.alpha
        batchSize = self.batchSize
        train_part = self.train_part
        threshold = self.threshold
        regular = self.regular
        attenuation = self.attenuation

        self.grpcclient = modelClientApi.GrpcClient("logisticModel.yaml")  # ?????????
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # ????????????
        self.linr_GeneratePK()
        ExPrint.extdebug('??????grpc??????')

        # ???????????????HOST ??????????????? ??????csv ??????????????????????????????jobid?????????????????????

        # 1.1 ??????host ?????????????????????
        tabledict = {}
        tabledict['jobid'] = self.jobid
        tabledict['name'] = self.name
        tabledict['start_time'] = self.start_time
        tabledict['hostcsv'] = self.hostcsv
        tabledict['hostcolum'] = self.hostcolum
        tabledict['regular'] = self.regular
        ExPrint.extdebug("model trancode=transfer CSV info start ...")
        uuid = self.jobid + self.name
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode='logistic_modelhostcsvinfo', uuid=uuid,
                                                  reqdata=tabledict))

        retcode = response.respcode
        retmsg = pickle.loads(response.respdata)
        ExPrint.extdebug(
            "model  trancode=modelhostcsvinfo  end ... retcode=" + callgrpcException(retcode, "modelhostcsvinfo"))
        # ExPrint.extdebug("model  trancode=modelhostcsvinfo  end ... retcode=" + str(retcode)+"retmsg="+retmsg)
        if retcode != 0:
            ExPrint.exterror(
                "model  trancode=modelhostcsvinfo  failed end ... retcode=" + str(retcode) + "retmsg=" + retmsg)
            exit(3)
        else:
            ExPrint.extdebug("model  trancode=modelhostcsvinfo  end !!")
        # load csv data into alldata
        icount = self.loadcsvdata()
        ExPrint.extdebug("loadcsvdata" + self.guestcsv + str(icount) + "end!")

        ''' shuffle data '''
        indexs = self.shuffle_data(self.alldata)

        self.alldata = self.alldata[indexs]
        req = self.grpcclient.request_from_OnetoOne(trancode='logistic_shuffle_data', uuid='uuid_shuffle_data',
                                                    reqdata=indexs)
        response = self.stub.OnetoOne(req)
        retcode = response.respcode

        ExPrint.extdebug("grpc   shuffle_data  fininshed" + callgrpcException(retcode, "shuffle_data"))
        GrpcFailedExit(response.respcode)

        '''   ???????????????????????????  '''
        ExPrint.extdebug("??????????????????????????? train_test_div  start " + datapath + self.name + self.jobid + '_guest.npz')
        self.data, self.testdata = logistic_basic.train_test(self.alldata, self.train_part)
        trainDateCount = str(np.shape(self.data)[0])
        testDateCount = str(np.shape(self.testdata)[0])
        ExPrint.saveJsondata("trainDateCount=" + trainDateCount)
        ExPrint.saveJsondata("testDateCount=" + testDateCount)
        self.dataname = datapath + self.name + self.jobid + '_guest.npz'
        np.savez(self.dataname, self.data, self.testdata)
        req = self.grpcclient.request_from_OnetoOne(trancode='logistic_train_test_div', uuid='uuid_train_test_div',
                                                    reqdata=train_part)
        response = self.stub.OnetoOne(req)
        ExPrint.extdebug("grpc   train_test_div  fininshed" + callgrpcException(response.respcode, "train_test_div"))
        GrpcFailedExit(response.respcode)
        useall, indexs = logistic_basic.get_random_sample(self.data, batchSize)
        batch_data = logistic_basic.get_index_array(self.data, useall, indexs)
        # print("data=",self.data)
        # print("testdata=",self.testdata)
        # print("useall=",useall, "idexs=",indexs)
        if useall == 1:
            req = self.grpcclient.request_from_OnetoOne(trancode='logistic_random_index_array', uuid='uuid_for_index',
                                                        reqdata=indexs)
            response = self.stub.OnetoOne(req)
            ExPrint.extdebug(
                "useall==1 grpc   logistic_random_index_array  fininshed" + callgrpcException(response.respcode,
                                                                                              "logistic_random_index_array"))
            GrpcFailedExit(response.respcode)
        m = np.shape(batch_data)[0]
        y = batch_data[:, 1].reshape(m, 1)
        x = batch_data[:, 2:]
        # print(f"x={m}y={y}x={x}")
        init_w_G = logistic_basic.init_w(x)
        # guest_w = init_w_G
        self.w = init_w_G
        wx_G = np.matmul(x, init_w_G)  # ???????????????wx_G

        # uuid = file.split('\\')[-1]#win\\???linux\
        uuid = self.jobid
        req = self.grpcclient.request_from_OnetoOne(trancode='logistic_remote_wx', uuid=uuid, reqdata='reqdata')
        response = self.stub.OnetoOne(req)
        wx_H = pickle.loads(response.respdata)  # ???????????????wx_H
        print(wx_H)
        ExPrint.extdebug("logistic_remote_wx" + callgrpcException(response.respcode, "logistic_remote_wx"))
        GrpcFailedExit(response.respcode)
        m = len(wx_H)
        wx_H = np.array(wx_H).reshape(m, 1)  # ???????????????
        converged.set_threshold(threshold)  # ??????????????????????????????????????????????????????????????????
        self.model_name = model_path + self.model_name
        # ????????????
        ExPrint.extdebug("???????????????????????????  start ...")
        pv = f"epoch={self.epoch} alpha={alpha} batchSize={self.batchSize}train_part={self.train_part}threshold={self.threshold}regular={self.regular}attenuation={attenuation}"
        ExPrint.extdebug(pv)
        div_g_mean = init_w_G * 0
        for i in range(self.epoch):
            alpha = alpha * attenuation  ### ?????????????????????alpha?????????
            residual = logistic_basic.compute_d(wx_G, wx_H, y)  # ????????????
            lst_residual = list(map(lambda x: x[0], residual.tolist()))
            loss = logistic_basic.jtheta(residual)  # ??????loss
            converged.mean_10(loss=loss)
            '''????????????????????????????????????????????????  ?????????????????????????????????????????????'''
            if converged.logistic_stop_or_ahead() or i == self.epoch - 1:  # ????????????
                # ????????????
                reqdata = self.name + self.jobid
                req = self.grpcclient.request_from_OnetoOne(trancode='logistic_model_save', uuid='uuid_model_save',
                                                            reqdata=reqdata)
                response = self.stub.OnetoOne(req)
                retmsg = pickle.loads(response.respdata)
                model_save(self.model_name, parameter=self.w)
                model_save_upgrade({"method": "logistic", "para": self.w, "column": self.guestcolum}, self.jobid,
                                   method='train_guest')
                ExPrint.extdebug("?????????????????????????????????????????????????????? ????????????????????????" + self.model_name + "end!")
                GrpcFailedExit(response.respcode)
                break
            m, n = np.shape(batch_data)
            div_j = logistic_basic.gradient_regul(x, residual, self.regular, 0.00001, self.w, m)  # ????????????
            # div_j = logistic_basic.gradient( residual,x)  # ????????????
            # div_g_mean = 0.9 * div_g_mean + 0.1 * div_j
            div_g_mean = get_mean(div_g_mean, div_j, 10)
            self.w = logistic_basic.update_w(self.w, div_g_mean, alpha)  # ??????w_G
            print(self.w)

            # print(np.shape(wx_G))

            '''
            ??????GRPC??????
            ??????????????????????????????residualE?????????????????????div_j_H
            '''
            # ExPrint.extdebug(" ????????????residualE start !")
            ## residualE = list(map(lambda x: self.public_key.encrypt(x), lst_residual))#????????????
            # ExPrint.extdebug(" ????????????residualE finished!")
            # with Pool(20) as pool:
            #     residualE = pool.map(self.public_key.encrypt,lst_residual)
            # print("jiamijiami",residualE)
            #
            uname = 'logistic_get_residual:'
            unum = str(i)
            uuid = uname + unum
            req = self.grpcclient.request_from_OnetoOne(trancode='logistic_get_residual', uuid=uuid,
                                                        reqdata=residual)  # ????????????residual
            response = self.stub.OnetoOne(req)
            div_j_H = pickle.loads(response.respdata)  # ??????host???????????????
            # print("list_residual=",lst_residual)
            ExPrint.extdebug(" for logistic_get_residual " + uuid + "finished!")
            GrpcFailedExit(response.respcode)

            '''???1.5????????????batch_data'''
            useall, indexs = logistic_basic.get_random_sample(self.data, batchSize)
            batch_data = logistic_basic.get_index_array(self.data, useall, indexs)
            if useall == 1:
                req = self.grpcclient.request_from_OnetoOne(trancode='logistic_random_index_array',
                                                            uuid='uuid_for_index',
                                                            reqdata=indexs)
                response = self.stub.OnetoOne(req)
                ExPrint.extdebug("for logistic_random_index_array " + callgrpcException(response.respcode,
                                                                                        "logistic_random_index_array"))
                GrpcFailedExit(response.respcode)
            m = np.shape(batch_data)[0]
            y = batch_data[:, 1].reshape(m, 1)
            x = batch_data[:, 2:]
            wx_G = logistic_basic.wx(x, self.w)
            print("div_j_H=", div_j_H)
            '''????????????????????????????????????????????????host?????????????????????
            '''
            # with  Pool() as pool:
            #     div_j_H = pool.map(self.privacy_key.decrypt, div_j_H)

            ## div_j_H = list(map(lambda x: self.privacy_key.decrypt(x), div_j_H))

            reqdata = []
            reqdata.append(div_j_H)
            reqdata.append(alpha)
            uname = 'linr_div_j:'
            unum = str(i)
            uuid = uname + unum
            req = self.grpcclient.request_from_OnetoOne(trancode='logistic_req_div_j', uuid=uuid, reqdata=reqdata)
            response = self.stub.OnetoOne(req)
            wx_H = pickle.loads(response.respdata)  # ??????????????????wx_H
            m = len(wx_H)
            wx_H = np.array(wx_H).reshape(m, 1)  # ???????????????
            ExPrint.extdebug("for logistic_req_div_j " + uuid + "finished!" + callgrpcException(response.respcode,
                                                                                                "logistic_req_div_j"))
            GrpcFailedExit(response.respcode)
        model_test_result=self.model_test()
        ExPrint.saveJsondata("model_data=" + str(model_test_result))
        sqlc = mysqlClass()
        sql = 'update smpc_model set model_status=%s,model_evaluate=%s where id=%s '
        sqlc._execute(sql,param=['06',str(model_test_result),self.jobid])


# get  linear Mode  path param
def getlinearModePathPara(para=None):
    if para is None:
        loadyaml = load_yaml_conf("gobal_conf.yaml")
    else:
        loadyaml = load_yaml_conf(para)
    return getDirectValue(loadyaml, 'guest_modelpath'), getDirectValue(loadyaml, 'guest_datapath')
    # ????????????
    # def save_model(self,):


if __name__ == '__main__':
    #     jobid = "1405343244878561282"
    #     name = 'logistic'
    #     start_time = 'args.start_time'
    #     guesttable = "/data/zhanghui/woe/logistic_y.csv"
    #     hosttable = "/data/zhanghui/woe/logistic_x.csv"
    #     guestcolum = ['id', 'y', 'x1', 'x2']
    #     hostcolum = ['id', 'x0', 'x1', 'x2', 'x3']
    #
    #     # guesttable = "/data/zhanghui/woe/y_woe.csv"
    #     # hosttable = "/data/zhanghui/woe/x_woe.csv"
    #     # guestcolum = ['id','x1','x2','x3','x4','x5']
    #     # hostcolum = ['id','x4','x5','x6','x7']
    #
    #     getoutfile = "/data/zhanghui/woe/y_logistic.csv"
    #     hostoutfile = "/data/zhanghui/woe/x_logistic.csv"
    #     boxCount = 5
    #     boxMethod = 'avg'
    #     epoch = 2000
    #     alpha = 0.5
    #     batchSize = 2000
    #     train_part = 0.9
    #     threshold = 0.000001
    #     regular = 'no'
    #     attenuation = 1
    #     outfilename = "/data/zhanghui/python/federatedml/linear/log/outfile.npy"
    #     # print("v={}{}{}{}{}{}{}".format(jobid,name,start_time,guesttable,hosttable,guestcolum,hostcolum))
    #     ExPrint.extdebug("lin_mode  Start ...")
    #     ExPrint.extdebug("get json data start ....")
    #     logistic_guest = logistic_guest()
    #
    #     logistic_guest.assign_init_csvpara(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum,
    #                                        outfilename)
    #     logistic_guest.assign_model_param_value(epoch, alpha, batchSize, train_part, threshold, regular, attenuation)
    #     # send host information
    #     logistic_guest.fit()
    #     print("GGGGGGGGGGGGG")
    # '''
    # if __name__ == '__main__':
    #     #security_level, block_num, random_bit = readbasesecureinfo("appconf.yaml")
    #     file = "F:\\datasource\\linr_data_y.csv"
    #     lg=linear_guest()
    #     lg.loaddata(file)
    #     lg.fit(1000,0.0001,0.5,1,"model1",r'log/',threshold=0.1)
    #     lg.model_test("model1",r'log/')
    #     '''

    parser = argparse.ArgumentParser(description="secureinfo")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-start_time", "--start_time", type=str, required=True)
    parser.add_argument("-guestcsv", "--guestcsv", type=str, required=True)
    parser.add_argument("-hostcsv", "--hostcsv", type=str, required=True)
    parser.add_argument("-guestcsv_column_name", "--guestcsv_column_name", type=str, required=True)
    parser.add_argument("-hostcsv_column_name", "--hostcsv_column_name", type=str, required=True)
    # parser.add_argument("-modelResult", "--modelResult", type=str, required=False)
    parser.add_argument("-epoch", "--epoch", type=str, required=True)
    parser.add_argument("-alpha", "--alpha", type=str, required=True)
    parser.add_argument("-batchSize", "--batchSize", type=str, required=True)
    parser.add_argument("-train_part", "--train_part", type=str, required=True)
    parser.add_argument("-threshold", "--threshold", type=str, required=True)
    parser.add_argument("-regular", "--regular", type=str, required=True)
    parser.add_argument("-attenuation", "--attenuation", type=str, required=True)
    args = parser.parse_args()

    jobid = args.id
    name = args.name
    start_time = args.start_time
    guesttable = args.guestcsv
    hosttable = args.hostcsv
    guestcolum = args.guestcsv_column_name
    hostcolum = args.hostcsv_column_name

    # outfilename = args.modelResult
    epoch = args.epoch
    alpha = args.alpha
    batchSize = args.batchSize
    train_part = args.train_part
    threshold = args.threshold
    regular = args.regular
    attenuation = args.attenuation

    # print("v={}{}{}{}{}{}{}".format(jobid,name,start_time,guesttable,hosttable,guestcolum,hostcolum))
    ExPrint.extdebug("lin_mode  Start ...")
    ExPrint.extdebug("get json data start ....")
    logistic_guest = logistic_guest()
    logistic_guest.assign_init_csvpara(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum)
    logistic_guest.assign_model_param_value(epoch, alpha, batchSize, train_part, threshold, regular, attenuation)
    # send host information
    logistic_guest.fit()
    print("GGGGGGGGGGGGG")
'''
if __name__ == '__main__':
    #security_level, block_num, random_bit = readbasesecureinfo("appconf.yaml")
    file = "F:\\datasource\\linr_data_y.csv"
    lg=linear_guest()
    lg.loaddata(file)
    lg.fit(1000,0.0001,0.5,1,"model1",r'log/',threshold=0.1)
    lg.model_test("model1",r'log/')
    '''
