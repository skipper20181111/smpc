# -*-coding:utf-8-*-
import argparse
import ast
import os
from baseCommon.dataIO import loadcsvFiledata, dumpdfTocsvFile, loadcsvHead
from baseCommon.extprint import ExPrint
from baseCommon.pymysqlclass import mysqlClass
from baseInterface import modelClientApi
from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
import random
import pickle
from baseInterface.model_grpcException import GrpcExceptions, checkRetcode
from secureprotol import gmpy_math
from federatedml.intersect_statistic.intersect import RsaIntersect
import numpy as np
import time


def readbasesecureinfo(para=None):
    if para is None:
        loadyaml = load_yaml_conf("gobal_conf.yaml")
    else:
        loadyaml = load_yaml_conf(para)
        ptr = getDirectValue(loadyaml, "basesecureinfo")
        return getDirectValue(ptr, 'security_level'), getDirectValue(ptr, 'block_num'), getDirectValue(ptr,
                                                                                                       'random_bit')


class RsaIntersectGuest(RsaIntersect):
    def __init__(self):
        super().__init__()
        self.uuid = 'Intersect'
        self.e = None
        self.n = None
        self.stub = None
        self.r = random.SystemRandom().getrandbits(128)
        self.grpcclient = None
        self.jobid = None
        self.name = None
        self.jobtype = None
        self.start_time = None
        self.guestcsv = None
        self.hostcsv = None
        self.guestcolum = None
        self.hostcolum = None
        self.hostoutfilename = None
        self.guestoutfilename = None

    def ini_csvpara(self, jobid, name, jobtype, start_time, guestfile, hostfile, guestcolum, hostcolum,
                    guestoutfilename,
                    hostoutfilename):
        self.jobid = jobid
        self.name = name
        self.jobtype = jobtype
        self.start_time = start_time
        self.guestcsv = guestfile
        self.hostcsv = hostfile
        self.guestcolum = guestcolum
        self.hostcolum = hostcolum
        self.hostoutfilename = hostoutfilename
        self.guestoutfilename = guestoutfilename

    # ???????????????host?????????pk
    def pk_request(self):
        trancode = 'pkrequest'
        req = self.grpcclient.request_from_OnetoOne(trancode=trancode, uuid=self.uuid, reqdata="requestPk")
        response = self.stub.OnetoOne(req)
        if response.respcode != 0:
            return 500, pickle.loads(response.respdata)
        public_key = pickle.loads(response.respdata)
        ExPrint.extdebug(f"public_key={public_key}")
        self.e = public_key['e']
        self.n = public_key['n']
        return 0, "????????????????????????"

    # ??????guest_id,????????????G1
    def cal_g1(self, guest_id):
        g1 = gmpy_math.powmod(self.r, int(self.e), int(self.n)) * int(RsaIntersect.hash(guest_id), 16) % int(self.n)
        return g1

    def get_g2(self, guest_id):
        g1 = list(map(lambda x: [self.cal_g1(x)], guest_id))
        ExPrint.extdebug(f"??????g1??????????????????")
        g2_reserve = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode='g2request', uuid=self.uuid, reqdata=g1))  # [g1,g2]
        ExPrint.extdebug("???????????????????????????")
        if g2_reserve.respcode != 0:
            return 500, pickle.loads(g2_reserve.respdata)
        return 0, np.hstack((guest_id, pickle.loads(g2_reserve.respdata))).tolist()

    def cal_gf(self, g2):
        if isinstance(g2, list):
            g2 = g2[0]
        gf = RsaIntersect.hash(gmpy_math.divm(int(g2), int(self.r), int(self.n)))
        return gf

    # ??????HF
    def get_gf(self, g2):
        gf = list(map(self.cal_gf, g2))
        return gf

    # ??????HF
    def get_hf(self):
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode='hfrequest', uuid=self.uuid, reqdata='start calculate hf'))
        if response.respcode != 0:
            return 500, pickle.loads(response.respdata)
        HF = pickle.loads(response.respdata)
        return 0, HF

    # ????????????
    def send_intersect(self, data):
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode='send_intersect', uuid=self.uuid, reqdata=data))
        if response.respcode != 0:
            # raise RuntimeError(pickle.loads(response.respdata))
            return 500, pickle.loads(response.respdata)
        return 0, pickle.loads(response.respdata)

    def run(self, guest_data):
        start_time = time.time()
        if len(guest_data) == 0:
            ExPrint.extdebug("???????????????????????????")
            exit(999)

        guest_data = np.array(guest_data)
        guest_id = guest_data[:, [0]].tolist()

        self.grpcclient = modelClientApi.GrpcClient("intersect.yaml")
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)
        if not self.stub:
            ExPrint.extdebug("?????????????????????????????????")
            exit(999)

        # ??????host ?????????????????????
        tabledict = {}
        tabledict['jobid'] = self.jobid
        tabledict['name'] = self.name
        tabledict['jobType'] = self.jobtype
        tabledict['start_time'] = self.start_time
        tabledict['hostcsv'] = self.hostcsv
        print(f"shishish{self.hostcsv}")
        tabledict['hostcolum'] = self.hostcolum
        tabledict['hostoutfilename'] = self.hostoutfilename
        ExPrint.extdebug("trancode=transfer CSV info start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='hostparam', uuid=self.uuid,
                                                                            reqdata=tabledict))
        retcode = response.respcode
        if retcode != 0:
            ExPrint.extdebug("trancode = hostcsvinfo end ... retcode=" + str(retcode))
            exit(999)

        # ???????????????
        retcode, retmsg = self.pk_request()
        print("?????????????????????")
        ExPrint.extdebug("?????????????????????")
        retcodeexception = checkRetcode(retcode, retmsg)
        try:
            retcodeexception.raise_retcode()
        except GrpcExceptions:
            exit(999)

        # ??????guest_id?????????G2
        retcode, G2_reserve = self.get_g2(guest_id)
        ExPrint.extdebug("??????Guest_id,??????G2???")
        retcodeexception = checkRetcode(retcode, G2_reserve)
        try:
            retcodeexception.raise_retcode()
        except GrpcExceptions:
            exit(999)

        G2 = np.array(G2_reserve)[:, [2]].tolist()
        ExPrint.extdebug("G2????????????-??????G2?????????")
        # ??????G2?????????GF
        GF = self.get_gf(G2)
        ExPrint.extdebug("??????G2,??????GF???")
        GF = [[x] for x in GF]
        ExPrint.extdebug("GF?????????????????????")
        GF_reserve = np.hstack((G2_reserve, GF))  # [[value, G1,G2, GF],[...]]
        ExPrint.extdebug("??????GF?????????")
        # ??????HF
        retcode, HF = self.get_hf()
        ExPrint.extdebug("??????Host??????HF")
        retcodeexception = checkRetcode(retcode, HF)

        try:
            retcodeexception.raise_retcode()
        except GrpcExceptions as e:
            exit(999)
        ExPrint.extdebug("??????HF?????????")

        intersect_hash_ids = np.intersect1d(HF, GF)
        ExPrint.extdebug("??????????????????")

        # ???host??????id??????
        retcode, result = self.send_intersect(intersect_hash_ids)
        ExPrint.extdebug("???????????????Host?????????")

        retcodeexception = checkRetcode(retcode, result)
        try:
            retcodeexception.raise_retcode()
        except GrpcExceptions:
            exit(999)

        # ???numpy.intersect1d??????
        _, GF_reserve_ind, _ = np.intersect1d(GF_reserve[:, 3], intersect_hash_ids, return_indices=True)
        intersect_ids = GF_reserve[GF_reserve_ind, 0]

        ExPrint.extdebug("?????????????????????")

        # ???numpy.intersect1d??????
        _, guest_data_ind, _ = np.intersect1d(guest_data[:, 0], intersect_ids, return_indices=True)
        intersect_data = guest_data[guest_data_ind, :].tolist()

        ExPrint.extdebug("???????????????????????????")
        total_count = len(intersect_hash_ids)

        sqlc = mysqlClass()
        param = [total_count, jobid]
        if jobType == "train":
            sql = "update smpc_model set inter_sect_num = %s  where id = %s"
            sqlc._execute(sql, param=param)
            sqlc._close()
        elif jobType == "predict":
            sql = "update smpc_predict set inter_sect_num = %s where id = %s"
            sqlc._execute(sql, param=param)
            sqlc._close()
        end_time = time.time()
        total_time = end_time - start_time
        ExPrint.extdebug(f"total_count={total_count}")
        print(f'total_time={total_time}???')

        # ????????????????????????
        outPath = os.path.join(os.path.dirname(os.path.dirname(self.guestcsv)), self.jobid)
        if not os.path.exists(outPath):
            os.mkdir(outPath)
        self.guestoutfilename = os.path.join(outPath, self.guestoutfilename)
        dumpdfTocsvFile(filename=self.guestoutfilename, Head=self.table_columns, data=intersect_data)
        return 0

    def querycsvdata_to_list(self):
        print("dd=", self.guestcsv, self.guestcolum, type(self.guestcolum))
        self.table_columns = rsaintersectguest.querycsvhead()
        self.guestcolum = ast.literal_eval(self.guestcolum)
        self.table_columns = self.guestcolum + [x for x in self.table_columns if x not in self.guestcolum]
        retlist = loadcsvFiledata(self.guestcsv, resetcolumn_1=self.table_columns)
        return retlist

    def querycsvhead(self):
        return loadcsvHead(self.guestcsv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="secureinfo")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-jobType", "--jobType", type=str, required=True)
    parser.add_argument("-startTime", "--startTime", type=str, required=True)
    parser.add_argument("-guestDataName", "--guestDataName", type=str, required=True)
    parser.add_argument("-hostDataName", "--hostDataName", type=str, required=True)
    parser.add_argument("-guestColumnName", "--guestColumnName", type=str, required=True)
    parser.add_argument("-hostColumnName", "--hostColumnName", type=str, required=True)
    parser.add_argument("-guestOutfileName", "--guestOutfileName", type=str, required=True)
    parser.add_argument("-hostOutfileName", "--hostOutfileName", type=str, required=True)
    args = parser.parse_args()

    jobid = args.id
    name = args.name
    jobType = args.jobType
    startTime = args.startTime
    guestDataName = args.guestDataName
    hostDataName = args.hostDataName
    guestColumnName = args.guestColumnName
    hostColumnName = args.hostColumnName
    guestOutfileName = args.guestOutfileName
    hostOutfileName = args.hostOutfileName

    print(
        "v={}{}{}{}{}{}{}".format(jobid, name, jobType, startTime, guestDataName, hostDataName, guestColumnName,
                                  hostColumnName))
    ExPrint.extdebug("call RsaIntersect Start ...")
    rsaintersectguest = RsaIntersectGuest()
    rsaintersectguest.ini_csvpara(jobid, name, jobType, startTime, guestDataName, hostDataName, guestColumnName,
                                  hostColumnName,
                                  guestOutfileName, hostOutfileName)
    guest_data = rsaintersectguest.querycsvdata_to_list()  # ???CSV??????guest?????????
    ExPrint.extdebug("querytabledata_to_list ???csv???????????????????????? end ??? ")

    rsaintersectguest.run(guest_data)
