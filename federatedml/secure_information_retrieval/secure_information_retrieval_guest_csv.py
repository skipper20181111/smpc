import random
import time
import argparse
import json
from baseCommon.baseConvert import getDirectValue, convertStrToInt, convertIntToStr
from baseCommon.dataIO import loadcsvFiledata, dumpdfTocsvFile
from federatedml.secure_information_retrieval.base_secure_information_retrieval import BaseSecureInformationRetrieval, \
    CryptoExecutor
from secureprotol import gmpy_math
from secureprotol.pohlig_hellman_encryption import PohligHellmanCipherKey
from baseInterface import model_pb2_grpc, modelClientApi
from baseInterface.utilsApi import *
from baseCommon.extprint import ExPrint
from baseCommon.conf_yaml import load_yaml_conf
import pickle
import numpy as np
from computeApi.paralleCompute import *

# readbasesecureinfo param
##include：security_level、block_num、random_bit
security_level = None
block_num = None
random_bit = None


def readbasesecureinfo(para=None):
    if para is None:
        loadyaml = load_yaml_conf("gobal_conf.yaml")
    else:
        loadyaml = load_yaml_conf(para)
        ptr = getDirectValue(loadyaml, "basesecureinfo")
        return getDirectValue(ptr, 'security_level'), getDirectValue(ptr, 'block_num'), getDirectValue(ptr,
                                                                                                       'random_bit')


class SecureInformationRetrievalGuest(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalGuest, self).__init__()
        self.grpcclient = None
        self.stub = None
        self.uuid = 'SecureInformationRetrieval'
        self.block_num = block_num
        self.security_level = security_level
        print("param=", security_level)
        self.r = None
        self.random_bit = random_bit
        #  20210508 add
        self.jobid = None
        self.name = None
        self.start_time = None
        self.guestcsv = None
        self.hostcsv = None
        self.guestcolum = None
        self.hostcolum = None
        self.outfilename = None

    def ini_csvpara(self, jobid, name, start_time, gustfile, hostfile, guestcolum, hostcolum, outfilename):
        self.jobid = jobid
        self.name = name
        self.start_time = start_time
        self.guestcsv = gustfile
        self.hostcsv = hostfile
        self.guestcolum = guestcolum
        self.hostcolum = hostcolum
        self.outfilename = outfilename

    def _init_model(self):
        self.commutative_cipher = CryptoExecutor(PohligHellmanCipherKey.generate_key())

    # 1. 同步通信秘钥
    def sir_pk_transfer(self):
        self._init_model()
        # print(self.commutative_cipher.cipher_core.mod_base)
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode="sirpkinit", uuid=self.uuid,
                                                                            reqdata=self.commutative_cipher))
        # print(f"response={pickle.loads(response.respdata)}")
        self.commutative_cipher.init()
        # print(self.commutative_cipher.cipher_core.exponent)

    def _encrypt_id(self, host_data, reserve_original_key=False):
        if reserve_original_key:
            return self.commutative_cipher.map_encrypt(host_data, mode=0)
        else:
            return self.commutative_cipher.map_encrypt(host_data, mode=1)

    def _decrypt_id_list(self, id_list):
        return self.commutative_cipher.map_decrypt(id_list, mode=2)

    def find_intersection(self, id_list_guest_second, id_list_host_second_only):
        id_list_intersect, key = cmp(id_list_guest_second, id_list_host_second_only)
        # id_list_intersect = []
        # for i in np.array(id_list_guest_second)[:,0]:
        #     # print(i)
        #     for j in np.array(id_list_host_second_only)[:,0]:
        #         if i == j:
        #             id_list_intersect.append([i,-1])
        # print(f"id_list_intersect={id_list_intersect}")
        # print(f"id_list_intersect1={id_list_intersect1}")
        # print(f"key={key}")
        # print(id_list_intersect==id_list_intersect1)
        return id_list_intersect

    def fake_blocks(self, id_list_intersect, id_list_host, replacement=True):
        intersect_count = len(id_list_intersect)
        print(f"intersect_count={intersect_count} block_num={self.block_num} ")
        self.target_block_index = random.randint(0, self.block_num - 1)
        # print(f"self.target_block_index= {self.target_block_index}")
        id_blocks = [None for _ in range(self.block_num)]
        # print(f"id_blocks={id_blocks}")
        for i in range(self.block_num):
            time_start = time.time()
            if i == self.target_block_index:
                id_block = id_list_intersect
                # print(f"第{i}次，目标时间={time.time()-time_start}")
            else:
                id_block = self.take_exact_sample(data_inst=id_list_host, exact_num=intersect_count)
                # print(f"第{i}次，取样时间={time.time() - time_start}")
            if not replacement:
                id_list_host, dict = cmp_diff(id_list_host, id_block)
                # print(f"第{i}次，求减时间={time.time() - time_start}")
                # id_list_host = self.subtract(id_list_host,id_block)
            time_en = time.time()
            id_block = self._decrypt_id_list(id_block)
            # print(f"第{i}次，加密时间={time.time() - time_en}")
            # print(f"id_block={id_block}")
            id_blocks[i] = id_block
        return id_blocks

    @staticmethod
    def take_exact_sample(data_inst, exact_num):
        row_rand_array = np.arange(len(data_inst))
        np.random.shuffle(row_rand_array)
        sample_inst = np.array(data_inst)[row_rand_array[0:exact_num]].tolist()
        return sample_inst

    @staticmethod
    def subtract(id_list_host, id_block):
        for i in np.array(id_list_host)[:, 0]:
            for j in np.array(id_block)[:, 0]:
                if i == j:
                    np.delete(id_list_host, i, axis=0)
        return id_list_host

    def run(self, guest_data):
        # guest_data [[1,2][2,3]]
        print("col=", self.hostcolum)
        ExPrint.extdebug("grpcclient start ...")
        self.grpcclient = modelClientApi.GrpcClient("secureinformatretrival.yaml")
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)
        # guest_data = np.array(guest_data)         # 转为numpy

        ExPrint.extdebug("grpcclient end ...")
        # print(guest_data)

        # 1. 同步通讯秘钥共有知识
        self.sir_pk_transfer()
        ExPrint.extdebug("sir_pk_transfer finished ...")

        # 1.1 发送host 的表相关的结构
        tabledict = {}
        tabledict['jobid'] = self.jobid
        tabledict['name'] = self.name
        tabledict['start_time'] = self.start_time
        tabledict['hostcsv'] = self.hostcsv
        tabledict['hostcolum'] = self.hostcolum
        ExPrint.extdebug("trancode=transfer CSV info start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='hostcsvinfo', uuid=self.uuid,
                                                                            reqdata=tabledict))
        retcode = pickle.loads(response.respdata)
        ExPrint.extdebug("trancode=hostcsvinfo  end ... retcode=" + retcode)
        # 2. 加密guest_id,得到Eg，发送，得到Eh

        ExPrint.extdebug("_encrypt_id  guest_data  start ...")
        id_list_guest_first = self._encrypt_id(guest_data, reserve_original_key=True)  # [g,Eg]
        # print(f'id_list_guest_first={id_list_guest_first}') #
        id_list_guest_first_only = list(map(lambda x: [x[1], -1], id_list_guest_first))
        # print(id_list_guest_first_only)
        ExPrint.extdebug("_encrypt_id  guest_data  end ...")
        ExPrint.extdebug("trancode=cal1stid start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal1stid', uuid=self.uuid,
                                                                            reqdata=id_list_guest_first_only))
        id_list_host_first = pickle.loads(response.respdata)  # [Eh,-1]
        ExPrint.extdebug("trancode=cal1stid end ...")
        # print(f"id_list_host_first={id_list_host_first}")

        # 3. 加密Eh，得到EEh，从host收到EEg

        ExPrint.extdebug("_encrypt_id  host  start ...")
        id_list_host_second = self._encrypt_id(id_list_host_first, reserve_original_key=True)  # [Eh,EEh]
        # print(f"id_list_host_second={id_list_host_second}")
        id_list_host_second_only = list(map(lambda x: [x[1], -1], id_list_host_second))
        ExPrint.extdebug("_encrypt_id  host  end ...")
        # print(f"id_list_host_second_only={id_list_host_second_only}")
        ExPrint.extdebug("trancode=cal2ndid start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal2ndid', uuid=self.uuid,
                                                                            reqdata="id_list_host_second_only"))
        id_list_guest_second = pickle.loads(response.respdata)  # [Eg,(EEg,-1)]
        ExPrint.extdebug("trancode=cal2ndid EEh end ...")

        t = 0
        # time1=time.time()
        # print(f"id_list_guest_first{id_list_guest_first}")
        # print(f"id_list_guest_second={id_list_guest_second}")
        # id_list_guest_second_reserve = cmp(id_list_guest_first, id_list_guest_second,1,0)
        # print(f"肖循环时间={time.time() - time1}")
        # print(f"肖id_list_guest_second_reserve={id_list_guest_second_reserve}")

        # [g,Eg]和[Eg,(EEg,-1)]生成[g,EEg],时间久，需要改进
        # id_list_guest_second_reserve = []
        # for i in id_list_guest_first:
        #     # print(t)
        #     for j in id_list_guest_second:
        #         if i[1] == j[0]:
        #             # print(i[1] == j[0])
        #             id_list_guest_second_reserve.append([i[0],j[1][0]])  # [g,EEg]
        # print(f"id_list_guest_second_reserve={id_list_guest_second_reserve}")
        ExPrint.extdebug("id_list_guest_second_reserve  start ...")
        id_list_guest_second_reserve = list(
            map(lambda x, y: [x[0], y[1][0]], id_list_guest_first, id_list_guest_second))

        # a=np.hstack((np.array(id_list_guest_first)[:,0],np.array(id_list_guest_second)[:,1]))

        # print(f"id_list_guest_second={id_list_guest_second}")

        ExPrint.extdebug("id_list_guest_second_reserve  end ...")
        id_list_guest_second_only = [(i[0], i[1]) for i in np.array(id_list_guest_second)[:, 1]]
        # print(f"id_list_guest_second_only={id_list_guest_second_only}")

        ExPrint.extdebug("id_list_guest_second_only  end ...")
        # 求交集
        # print(np.array(id_list_guest_second)[:,0])
        # print(np.array(id_list_host_second_only)[:,0])

        id_list_intersect = self.find_intersection(id_list_guest_second_only, id_list_host_second_only)  # [EEi,-1]
        ExPrint.extdebug("find_intersection EEi end ...")

        # print(f"id_list_intersect={id_list_intersect}")

        # 将intersect与guest id重建连接

        id_list_intersect_reserve, keys = cmp(id_list_intersect, id_list_guest_second_reserve, 0, 1)
        if len(id_list_intersect_reserve) == 0:
            return []
        ExPrint.extdebug(" cmp id_list_intersect,id_list_guest_second_reserve  (将intersect与guest id重建连接)   end ...")
        # id_list_intersect_reserve = []
        # for i in id_list_intersect:
        #     for j in id_list_guest_second_reserve:
        #         if i[0] == j[1]:
        #             id_list_intersect_reserve.append(j)

        # 需要优化
        # 制造传输数据集，发送给host

        id_blocks = self.fake_blocks(id_list_intersect, id_list_host_second_only)  # List[(EEi, -1)]
        time_fake_transfer = time.time()
        ExPrint.extdebug(" fake_blocks id_list_intersect,id_list_host_second_only  end ...")
        # print(f"id_blocks={id_blocks}")

        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="calreindexid", uuid=self.uuid, reqdata=id_blocks))  # 时间长

        ExPrint.extdebug(" grpc  trancode=calreindexid  (制造传输数据集，发送给host)  end ...")
        # print(pickle.loads(response.respdata))

        # 发起OT操作，获得秘钥的list

        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="executeot", uuid=self.uuid, reqdata="Request OT Keys"))
        key_list = pickle.loads(response.respdata)
        ExPrint.extdebug(" grpc  trancode=executeot (发起OT操作，获得秘钥的list)  end ...")
        # print(f"key_list={key_list}")

        # 生成随机数，用目标秘钥进行加密，发送给host，获取密文数据

        self.r = random.randint(2 ** (1024 - 1), key_list[self.target_block_index][1] - 1)
        # self.r = 10
        enc_r = gmpy_math.powmod(self.r, key_list[self.target_block_index][0], key_list[self.target_block_index][1])
        ExPrint.extdebug("gmpy_math.powmod (发起OT操作，获得秘钥的list)  end ...")
        # print(f"r={len(str(self.r))}")
        print(f"enc_r={len(str(enc_r))}  enc_{enc_r}")
        # print("开始报错")
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="getvalue", uuid=self.uuid, reqdata=enc_r))
        id_block_ciphertext = pickle.loads(response.respdata)
        ExPrint.extdebug(" grpc  trancode=getvalue (生成随机数，用目标秘钥进行加密，发送给host，获取密文数据)  end ...")

        # 将密文数据解密，获得查询结果
        # print(f"self.target_block_index={self.target_block_index}")
        # print(self.r)

        target_blocks_value = []
        for i in id_block_ciphertext[self.target_block_index]:
            # print(f"i={i[1:]}")
            target_block_value = list(map(lambda x: self.cal_divm(x), i[1:]))
            # print(target_block_value)
            # target_block =list(map(lambda x ,y: [x[0],y],id_list_intersect_reserve,target_block_value))
            target_blocks_value.append(target_block_value)
        ExPrint.extdebug("for id_block_ciphertext (将密文数据解密，获得查询结果)  end ...")
        target_blocks = np.hstack((np.array(id_list_intersect_reserve)[:, [0]], target_blocks_value)).tolist()
        ExPrint.extdebug("for hstack   end ...")
        print(f"target_blocks={target_blocks}")
        # print(len(target_blocks))
        retdata = convertIntToStr(target_blocks)
        # print(f"retdata={retdata}")
        ExPrint.extdebug("convertIntToStr  Finished ...")
        # create return filename
        # print("hostcolum=",self.hostcolum)
        retfile = dumpdfTocsvFile(self.outfilename, retdata, Head=self.hostcolum, index=False)
        ExPrint.extdebug("dumpdfTocsvFile  Finished ...")
        return retfile

    def cal_divm(self, value):
        # print(f"value={value}")
        # print(f"r={self.r}")
        # print(value>self.r)
        # value2 = value / self.r
        value2 = gmpy_math.divm(value, self.r, 2 ** 1024)
        return int(value2)

    def querycsvdata_to_list(self):
        print("dd=", self.guestcsv, self.guestcolum, type(self.guestcolum))
        retlist = loadcsvFiledata(self.guestcsv, resetcolumn_1=self.guestcolum)
        return retlist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="secureinfo")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-start_time", "--start_time", type=str, required=True)
    parser.add_argument("-guestcsv", "--guestcsv", type=str, required=True)
    parser.add_argument("-hostcsv", "--hostcsv", type=str, required=True)
    parser.add_argument("-guestcsv_column_name", "--guestcsv_column_name", type=str, required=True)
    parser.add_argument("-hostcsv_column_name", "--hostcsv_column_name", type=str, required=True)
    parser.add_argument("-outfilename", "--outfilename", type=str, required=True)
    args = parser.parse_args()

    jobid = args.id
    name = args.name
    start_time = args.start_time
    guesttable = args.guestcsv
    hosttable = args.hostcsv
    guestcolum = args.guestcsv_column_name
    hostcolum = args.hostcsv_column_name
    outfilename = args.outfilename

    print("v={}{}{}{}{}{}{}".format(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum))
    ExPrint.extdebug("call Secure_information_retrieval Start ...")
    security_level, block_num, random_bit = readbasesecureinfo("secureinformatretrival.yaml")
    secureinforguest = SecureInformationRetrievalGuest()
    secureinforguest.ini_csvpara(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum, outfilename)
    retlist = secureinforguest.querycsvdata_to_list()
    ExPrint.extdebug("querytabledata_to_list 从csv文件中取数据完成 end ！ ")
    guestid = convertStrToInt(retlist)
    ExPrint.extdebug("convertStrToInt 数据类型转换完成 end ！ ")
    secureinforguest.run(guestid)
    ##exit()
    # p=[]
    # for i in range(1):
    #    time_start = time.time()
    #    #guestid= np.arange(10**(i+2)).reshape(10**(i+2),1).tolist()
    #    #guestid=list(map(lambda  x: [int(x[0])],retlist))
    #    guestid =convertStrToInt(retlist)
    #    print(f"guestid={guestid}")
    #    #SecureInformationRetrievalGuest().run(guestid)
    #    secureinforguest.run(guestid)
    #    time_end = time.time()
    #    p.append(f"查询方数据量为={10**(i+2)},total_time={time_end-time_start}秒")
    # print(p)
    # total=100
    # i=4
    #
    # time_start = time.time()
    # guestid= np.arange(total).reshape(int(total/i),i).tolist()
    # # print(f"guestid={guestid}")
    # SecureInformationRetrievalGuest().run(guestid)
    # time_end = time.time()
