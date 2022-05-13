import argparse
import pickle
import random

from baseCommon.baseConvert import getDirectValue, convertStrToInt, convertIntToStr
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.extprint import ExPrint
from baseCommon.pymysqlclass import createsql, mysqlClass, RSPTYPE
from baseInterface import modelClientApi
from computeApi.paralleCompute import *
from federatedml.secure_information_retrieval.base_secure_information_retrieval import BaseSecureInformationRetrieval, \
    CryptoExecutor
from secureprotol import gmpy_math
from secureprotol.pohlig_hellman_encryption import PohligHellmanCipherKey

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
        self.jobid = None
        self.name = None
        self.start_time = None
        self.guesttable = None
        self.hosttable = None
        self.guestcolum = None
        self.hostcolum = None
        print("hostcolum={} hosttable={}".format(self.hostcolum, self.hosttable))

    def ini_tablepara(self, jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum):
        self.jobid = jobid
        self.name = name
        self.start_time = start_time
        self.guesttable = guesttable
        self.hosttable = hosttable
        self.guestcolum = guestcolum
        self.hostcolum = hostcolum

    def _init_model(self):
        self.commutative_cipher = CryptoExecutor(PohligHellmanCipherKey.generate_key())

    # 1. 同步通信秘钥
    def sir_pk_transfer(self):
        self._init_model()
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode="sirpkinit", uuid=self.uuid,
                                                                            reqdata=self.commutative_cipher))
        self.commutative_cipher.init()

    def _encrypt_id(self, host_data, reserve_original_key=False):
        if reserve_original_key:
            return self.commutative_cipher.map_encrypt(host_data, mode=0)
        else:
            return self.commutative_cipher.map_encrypt(host_data, mode=1)

    def _decrypt_id_list(self, id_list):
        return self.commutative_cipher.map_decrypt(id_list, mode=2)

    def find_intersection(self, id_list_guest_second, id_list_host_second_only):
        id_list_intersect, key = cmp(id_list_guest_second, id_list_host_second_only)
        return id_list_intersect

    def fake_blocks(self, id_list_intersect, id_list_host, replacement=True):
        intersect_count = len(id_list_intersect)
        print(f"intersect_count={intersect_count} block_num={self.block_num} ")
        self.target_block_index = random.randint(0, self.block_num - 1)
        id_blocks = [None for _ in range(self.block_num)]
        for i in range(self.block_num):
            if i == self.target_block_index:
                id_block = id_list_intersect
            else:
                id_block = self.take_exact_sample(data_inst=id_list_host, exact_num=intersect_count)
            if not replacement:
                id_list_host, dict = cmp_diff(id_list_host, id_block)
            id_block = self._decrypt_id_list(id_block)
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
        print("col=", self.hostcolum)
        ExPrint.extdebug("grpcclient start ...")
        self.grpcclient = modelClientApi.GrpcClient("secureinformatretrival.yaml")
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)
        ExPrint.extdebug("grpcclient end ...")

        # 1. 同步通讯秘钥共有知识
        self.sir_pk_transfer()
        ExPrint.extdebug("sir_pk_transfer finished ...")

        # 1.1 发送host 的表相关的结构
        tabledict = {}
        tabledict['jobid'] = self.jobid
        tabledict['name'] = self.name
        tabledict['start_time'] = self.start_time
        tabledict['hosttable'] = self.hosttable
        tabledict['hostcolum'] = self.hostcolum
        ExPrint.extdebug("trancode=transfer table info start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='hosttableinfo', uuid=self.uuid,
                                                                            reqdata=tabledict))
        retcode = pickle.loads(response.respdata)
        ExPrint.extdebug("trancode=hosttableinfo  end ... retcode=" + retcode)
        # 2. 加密guest_id,得到Eg，发送，得到Eh

        ExPrint.extdebug("_encrypt_id  guest_data  start ...")
        id_list_guest_first = self._encrypt_id(guest_data, reserve_original_key=True)  # [g,Eg]
        id_list_guest_first_only = list(map(lambda x: [x[1], -1], id_list_guest_first))
        ExPrint.extdebug("_encrypt_id  guest_data  end ...")
        ExPrint.extdebug("trancode=cal1stid start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal1stid', uuid=self.uuid,
                                                                            reqdata=id_list_guest_first_only))
        id_list_host_first = pickle.loads(response.respdata)  # [Eh,-1]
        ExPrint.extdebug("trancode=cal1stid end ...")

        # 3. 加密Eh，得到EEh，从host收到EEg
        ExPrint.extdebug("_encrypt_id  host  start ...")
        id_list_host_second = self._encrypt_id(id_list_host_first, reserve_original_key=True)  # [Eh,EEh]
        # print(f"id_list_host_second={id_list_host_second}")
        id_list_host_second_only = list(map(lambda x: [x[1], -1], id_list_host_second))
        ExPrint.extdebug("_encrypt_id  host  end ...")
        ExPrint.extdebug("trancode=cal2ndid start ...")
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal2ndid', uuid=self.uuid,
                                                                            reqdata="id_list_host_second_only"))
        id_list_guest_second = pickle.loads(response.respdata)  # [Eg,(EEg,-1)]
        ExPrint.extdebug("trancode=cal2ndid EEh end ...")

        ExPrint.extdebug("id_list_guest_second_reserve  start ...")
        id_list_guest_second_reserve = list(
            map(lambda x, y: [x[0], y[1][0]], id_list_guest_first, id_list_guest_second))

        ExPrint.extdebug("id_list_guest_second_reserve  end ...")
        id_list_guest_second_only = [(i[0], i[1]) for i in np.array(id_list_guest_second)[:, 1]]

        ExPrint.extdebug("id_list_guest_second_only  end ...")
        # 求交集
        id_list_intersect = self.find_intersection(id_list_guest_second_only, id_list_host_second_only)  # [EEi,-1]
        ExPrint.extdebug("find_intersection EEi end ...")

        # 将intersect与guest id重建连接
        id_list_intersect_reserve, keys = cmp(id_list_intersect, id_list_guest_second_reserve, 0, 1)
        if len(id_list_intersect_reserve) == 0:
            return []
        ExPrint.extdebug(" cmp id_list_intersect,id_list_guest_second_reserve  (将intersect与guest id重建连接)   end ...")
        # 制造传输数据集，发送给host

        id_blocks = self.fake_blocks(id_list_intersect, id_list_host_second_only)  # List[(EEi, -1)]
        time_fake_transfer = time.time()
        ExPrint.extdebug(" fake_blocks id_list_intersect,id_list_host_second_only  end ...")
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="calreindexid", uuid=self.uuid, reqdata=id_blocks))  # 时间长

        ExPrint.extdebug(" grpc  trancode=calreindexid  (制造传输数据集，发送给host)  end ...")
        # 发起OT操作，获得秘钥的list
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="executeot", uuid=self.uuid, reqdata="Request OT Keys"))
        key_list = pickle.loads(response.respdata)
        ExPrint.extdebug(" grpc  trancode=executeot (发起OT操作，获得秘钥的list)  end ...")
        # 生成随机数，用目标秘钥进行加密，发送给host，获取密文数据
        self.r = random.randint(2 ** (1024 - 1), key_list[self.target_block_index][1] - 1)
        enc_r = gmpy_math.powmod(self.r, key_list[self.target_block_index][0], key_list[self.target_block_index][1])
        ExPrint.extdebug("gmpy_math.powmod (发起OT操作，获得秘钥的list)  end ...")
        print(f"enc_r={len(str(enc_r))}  enc_{enc_r}")
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="getvalue", uuid=self.uuid, reqdata=enc_r))
        id_block_ciphertext = pickle.loads(response.respdata)
        ExPrint.extdebug(" grpc  trancode=getvalue (生成随机数，用目标秘钥进行加密，发送给host，获取密文数据)  end ...")

        # 将密文数据解密，获得查询结果
        target_blocks_value = []
        for i in id_block_ciphertext[self.target_block_index]:
            target_block_value = list(map(lambda x: self.cal_divm(x), i[1:]))
            target_blocks_value.append(target_block_value)
        ExPrint.extdebug("for id_block_ciphertext (将密文数据解密，获得查询结果)  end ...")
        target_blocks = np.hstack((np.array(id_list_intersect_reserve)[:, [0]], target_blocks_value)).tolist()
        ExPrint.extdebug("for hstack   end ...")
        print(f"target_blocks={target_blocks}")
        retdata = convertIntToStr(target_blocks)
        print(f"retdata={retdata}")
        ExPrint.extdebug("convertIntToStr  Finished ...")
        return retdata

    def cal_divm(self, value):
        value2 = gmpy_math.divm(value, self.r, 2 ** 1024)
        return int(value2)

    def querytabledata_to_list(self):
        print("dd=", self.guesttable, self.guestcolum)
        selectsql = createsql(self.guesttable, self.guestcolum)
        mysqlclass = mysqlClass()
        print("sql=", selectsql)
        retlist = mysqlclass._fetchall(selectsql, RSPTYPE.LIST)
        print("retlist=", retlist)
        mysqlclass._close()
        return retlist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="secureinfo")
    parser.add_argument("-id", "--id", type=str, required=True)
    parser.add_argument("-name", "--name", type=str, required=True)
    parser.add_argument("-start_time", "--start_time", type=str, required=True)
    parser.add_argument("-guest_table_name", "--guest_table_name", type=str, required=True)
    parser.add_argument("-host_table_name", "--host_table_name", type=str, required=True)
    parser.add_argument("-guest_column_name", "--guest_column_name", type=str, required=True)
    parser.add_argument("-host_column_name", "--host_column_name", type=str, required=True)
    args = parser.parse_args()

    jobid = args.id
    name = args.name
    start_time = args.start_time
    guesttable = args.guest_table_name
    hosttable = args.host_table_name
    guestcolum = args.guest_column_name
    hostcolum = args.host_column_name

    print("v={}{}{}{}{}{}{}".format(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum))
    ExPrint.extdebug("call Secure_information_retrieval Start ...")
    security_level, block_num, random_bit = readbasesecureinfo("secureinformatretrival.yaml")
    secureinforguest = SecureInformationRetrievalGuest()
    secureinforguest.ini_tablepara(jobid, name, start_time, guesttable, hosttable, guestcolum, hostcolum)
    retlist = secureinforguest.querytabledata_to_list()
    ExPrint.extdebug("querytabledata_to_list 从数据库表去数据完成 end ！ ")
    guestid = convertStrToInt(retlist)
    ExPrint.extdebug("convertStrToInt 数据类型转换完成 end ！ ")
    secureinforguest.run(guestid)
