import ast
import os
import time
from baseCommon.dataIO import loadcsvFiledata, dumpdfTocsvFile, loadcsvHead
from baseCommon.logger import LogClass, ON
from baseCommon.pymysqlclass import RSPTYPE, mysqlClass, createsql
from secureprotol.encrypt import RsaEncrypt
from secureprotol import gmpy_math
from federatedml.intersect_statistic.intersect import RsaIntersect
import multiprocessing as mp
import numpy as np

logclass = LogClass("testlog.txt")
LOGGER = logclass.get_Logger("levelname!!", ON.DEBUG)


class RsaIntersectHost(RsaIntersect):
    def __init__(self):
        super().__init__()
        self.uuid = None
        self.e = None
        self.d = None
        self.n = None
        self.jobType = None
        self.jobid = None
        self.name = None
        self.start_time = None
        self.hosttable = None
        self.hostcolum = None
        self.hostcsv = None

    def queryhostcsvdata_to_list(self, column_name=None):
        # 调用Host的Java，获取数据集路径
        self.table_columns = self.queryhostcsvhead()
        self.hostcolum = ast.literal_eval(self.hostcolum)
        self.table_columns = self.hostcolum + [x for x in self.table_columns if x not in self.hostcolum]
        retlist = loadcsvFiledata(self.hostcsv, columnName_1=column_name, resetcolumn_1=self.table_columns)
        return retlist

    def queryhostcsvhead(self):
        return loadcsvHead(self.hostcsv)

    def generate_pk(self, tradecode, uuid, reqdata):
        print(f"t={tradecode} uuid{uuid} data={reqdata}")
        if tradecode == 'pkrequest':
            self.uuid = uuid
            encrypt_operator = RsaEncrypt()
            encrypt_operator.generate_key()
            self.e, self.d, self.n = encrypt_operator.get_key_pair()
            public_key = {'e': self.e, 'n': self.n}
            return 0, public_key
        else:
            print("PkRequest Generate Error!")
            return 500, "PkRequest Generate Error!"

    def cal_g2(self, g1):
        if isinstance(g1, list):
            g1 = g1[0]
        g2 = gmpy_math.powmod(g1, self.d, self.n)
        return [g2]

    def queryhosttabledata_to_list(self):
        selectsql = createsql(self.hosttable, self.hostcolum)
        mysqlclass = mysqlClass()
        retlist = mysqlclass._fetchall(selectsql, RSPTYPE.LIST)
        mysqlclass._close()
        return retlist

    def request_g2(self, tradecode, uuid, reqdata):
        if tradecode == "g2request":
            print("开始连接数据库，查询文件全路径")
            sqlc = mysqlClass()
            print("数据库对象生成")
            param = [self.hostcsv]
            print(f"param={param}")
            if self.jobType == "train":
                print("开始查询数据")
                sql = "select file_path from smpc_data_info where data_name = %s"

                self.hostcsv = sqlc._fetchone(sql, param=param)[0]
                print(f"数据文件全路径为：{self.hostcsv}")
                sqlc._close()
            elif self.jobType == "predict":
                print("开始查询数据")
                sql = "select file_path from smpc_data_info where data_name = %s"
                self.hostcsv = sqlc._fetchone(sql, param=param)[0]
                print(f"数据文件全路径为：{self.hostcsv}")
                sqlc._close()
            print(f"HostCsvPath = {self.hostcsv}")
            print("start cal g2\n")
            time_1 = time.time()
            if self.hostcsv is not None:
                retlist = self.queryhostcsvdata_to_list()
            time_2 = time.time()
            print("Host Data input finished\n", time_2 - time_1)
            self.host_data = retlist
            if len(self.host_data) == 0:
                return 500, "Host数据导入错误！"
            self.host_id = np.array(self.host_data)[:, [0]].tolist()
            time_3 = time.time()
            print("Get Host Id List\n", time_3 - time_2)

            g1 = reqdata
            with mp.Pool() as pool:
                g2 = list(pool.map(self.cal_g2, g1))
            time_4 = time.time()
            print("Finish Calculate G2\n", time_4 - time_3)

            g2_reserve = np.hstack((g1, g2))  # [g1,g2]
            time_5 = time.time()
            print("Join G2 list to raw data\n", time_5 - time_4)
            return 0, g2_reserve
        else:
            print("G2 is not calculated!")
            return 500, "G2 is not calculated!"

    def cal_hf(self, host_id):
        HF = RsaIntersect.hash(gmpy_math.powmod(int(RsaIntersect.hash(host_id), 16), self.d, self.n))
        return HF

    def request_hf(self, tradecode, uuid, reqdata):
        if tradecode == "hfrequest":
            with mp.Pool() as pool:
                HF = list(pool.map(self.cal_hf, self.host_id))
            HF = [[x] for x in HF]
            self.HF_reserve = np.hstack((self.host_id, HF)).tolist()  # [id,HF]
            return 0, HF
        else:
            LOGGER.info("HF is not calculated!")
            return 500, "HF is not calculated!"

    def func_hostcsvinfo(self, trancode, uuid, reqdata):
        if trancode == "hostparam":
            hosttabledict = reqdata
            self.jobid = hosttabledict['jobid']
            self.name = hosttabledict['name']
            self.jobType = hosttabledict['jobType']
            self.start_time = hosttabledict['start_time']
            self.hostcsv = hosttabledict['hostcsv']
            self.hostcolum = hosttabledict['hostcolum']
            self.hostoutfilename = hosttabledict['hostoutfilename']
            print("参数传输结束！")
            return 0, "success"
        else:
            print("参数传输失败！")
            return 500, "参数传输失败！"

    def cal_intersect(self, trancode, uuid, reqdata):
        if trancode == "send_intersect":
            intersect = reqdata
            intersect = np.array(intersect)
            HF_reserve = np.array(self.HF_reserve)
            _, HF_reserve_ind, _ = np.intersect1d(HF_reserve[:, 1], intersect, return_indices=True)
            intersect_ids = HF_reserve[HF_reserve_ind, 0].tolist()
            host_data = np.array(self.host_data)
            _, host_data_ind, _ = np.intersect1d(host_data[:, 0], intersect_ids, return_indices=True)
            intersect_data = host_data[host_data_ind, :].tolist()

            outPath = os.path.join(os.path.dirname(os.path.dirname(self.hostcsv)), self.jobid)
            if not os.path.exists(outPath):
                os.mkdir(outPath)
            self.hostoutfilename = os.path.join(outPath, self.hostoutfilename)

            dumpdfTocsvFile(filename=self.hostoutfilename, Head=self.table_columns, data=intersect_data)
            return 0, "success"
        else:
            return 500, "Host接收交集失败！"
