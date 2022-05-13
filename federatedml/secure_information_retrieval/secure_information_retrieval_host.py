from baseCommon.baseConvert import convertStrToInt
from baseCommon.dataIO import loadcsvFiledata
from baseCommon.pymysqlclass import createsql, mysqlClass, RSPTYPE
from federatedml.intersect_statistic import get_data_from_db
from federatedml.secure_information_retrieval.base_secure_information_retrieval import BaseSecureInformationRetrieval, \
    CryptoExecutor
from secureprotol import gmpy_math
from secureprotol.pohlig_hellman_encryption import PohligHellmanCipherKey
import numpy as np
from secureprotol.encrypt import RsaEncrypt
from computeApi.paralleCompute import *


class SecureInformationRetrievalHost(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalHost, self).__init__()
        self.uuid = None
        # self.host_data = get_data_from_db.getdata(role="client")
        # self.host_data = np.arange(100).reshape(10,10).tolist()
        # print(f"self.host_data={self.host_data}")
        self.host_data = None
        self.id_list_host_first = None
        self.id_list_host_first_only = None
        self.id_list_guest_first = None
        self.id_list_guest_second = None
        self.block_num = 10
        self.public_key_list = []
        self.private_key_list = []
        self.id_blocks = None
        # 2021 add for hosttable
        self.jobid = None
        self.name = None
        self.start_time = None
        self.hosttable = None
        self.hostcolum = None
        self.hostcsv = None

    def _init_model(self):
        pass

    def fit(self, host_data):
        pass
        # 初始化通讯秘钥

    def sir_pk_init(self, tradecode, uuid, reqdata):
        if tradecode == "sirpkinit":
            self.commutative_cipher = reqdata
            self.commutative_cipher.init()
        self.uuid = uuid
        return 0, 'success'

    def _encrypt_id(self, host_data, reserve_value=False):
        if reserve_value:
            return self.commutative_cipher.map_encrypt(host_data, mode=3)
        else:
            return self.commutative_cipher.map_encrypt(host_data, mode=1)

    def queryhosttabledata_to_list(self):
        # print("dd=", self.guesttable, self.guestcolum)
        selectsql = createsql(self.hosttable, self.hostcolum)
        mysqlclass = mysqlClass()
        print("sql=", selectsql)
        retlist = mysqlclass._fetchall(selectsql, RSPTYPE.LIST)
        print("retlist=", retlist)
        mysqlclass._close()
        return retlist

    def queryhostcsvdata_to_list(self):
        retlist = loadcsvFiledata(self.hostcsv, resetcolumn_1=self.hostcolum)
        print("retlist=", retlist)
        return retlist

    def cal_first_id(self, trancode, uuid, reqdata):
        if trancode == 'cal1stid':
            if self.hosttable is not None:
                retlist = self.queryhosttabledata_to_list()
            if self.hostcsv is not None:
                retlist = self.queryhostcsvdata_to_list()
            self.host_data = convertStrToInt(retlist)
            # self.host_data = list(map(lambda x: [int(x[0])] , retlist))
            # self.host_data = list(map(lambda x: [int(x[0])] + [int(x[1])] + [int(x[2])], retlist))
            # self.host_data = list(map(lambda x: [int(x[0])]+[int(x[1])]+[int(x[2])], retlist))
            print("gg=", self.host_data)
            self.id_list_host_first = self._encrypt_id(self.host_data, reserve_value=True)  # [h, (Eh, Instance)]
            # print(f'id_list_host_first={self.id_list_host_first}')
            self.id_list_host_first_only = list(map(lambda x: [x[1][0], -1], self.id_list_host_first))  # (Eh, -1)
            # # print(f"id_list_host_first_only={self.id_list_host_first_only}")
            self.id_list_guest_first = reqdata
            # # print(f"id_list_guest_first={self.id_list_guest_first}")
            return 0, self.id_list_host_first_only

    def cal_2nd_id(self, trancode, uuid, reqdata):
        if trancode == "cal2ndid":
            self.id_list_guest_second = self._encrypt_id(self.id_list_guest_first, reserve_value=True)  # [Eg,EEg,-1]
            # # print(f"id_list_guest_second={self.id_list_guest_second}")
            return 0, self.id_list_guest_second

    @staticmethod
    def _restore_value(id_list_host, id_blocks):
        id_list_host_parse = list(map(lambda x: [x[1][0], x[1][1]], id_list_host))
        # print(f"id_list_host_parse={id_list_host_parse[0]}")
        # id_list_host_parse = id_list_host.map(lambda k, v: (v[0], v[1].label))     # (Eh, val)
        id_value_blocks = []
        for i in range(len(id_blocks)):
            # restored_table = []
            # for j in np.array(id_blocks[i])[:,0]:
            #     for k in id_list_host_parse:
            #         if j == k[0]:
            #             restored_table.append(k)
            restored_table, keys = cmp(id_blocks[i], id_list_host_parse)
            id_value_blocks.append(restored_table)
        return id_value_blocks

    # calreindexid
    def cal_reindex_id(self, trancode, uuid, reqdata):
        if trancode == "calreindexid":
            # print(f"reqdata={reqdata}")
            self.id_blocks = self._restore_value(self.id_list_host_first, reqdata)  # List[(Ei, val)]
            # print(f"id_blocks={self.id_blocks}")
        return 0, "success"

    def key_derivation(self, trancode, uuid, reqdata):
        if trancode == "executeot":
            for i in range(self.block_num):
                encrypt_operator = RsaEncrypt()
                encrypt_operator.generate_key()
                public_key = encrypt_operator.get_public_key()
                private_key = encrypt_operator.get_privacy_key()
                self.public_key_list.append(public_key)
                self.private_key_list.append(private_key)
            # print(f"public_key_list={self.public_key_list}")
            # print(f"private_key_list={self.private_key_list}")
            return 0, self.public_key_list

    def cal_value_cipher(self, trancode, uuid, reqdata):
        if trancode == "getvalue":
            dec_r = list(map(lambda x: gmpy_math.powmod(reqdata, x[0], x[1]), self.private_key_list))
            id_blocks_final = []
            for i in range(self.block_num):
                # print(dec_r[i])
                ## print(dec_r[i])
                ## print([(np.array(j) * dec_r[i] % (2**1024)).tolist() for j in np.array(self.id_blocks[i])[:,1]])
                id_block_id_final = np.array(self.id_blocks[i])[:, [0]].tolist()
                # print(f"id_block_id_final={id_block_id_final}")
                id_block_value_final = [(np.array(j) * dec_r[i] % (2 ** 1024)).tolist() for j in
                                        np.array(self.id_blocks[i])[:, 1]]
                id_block_final = np.hstack((id_block_id_final, id_block_value_final)).tolist()
                # id_block_final = list(map(lambda x,y:[x[0],y],id_block_id_final,id_block_value_final))
                # print(f"id_blocks_final={id_block_final}")
                id_blocks_final.append(id_block_final)

            return 0, id_blocks_final

        # calreindexid

    def func_hosttableinfo(self, trancode, uuid, reqdata):
        if trancode == "hosttableinfo":
            hosttabledict = reqdata
            print("hosttabledict=", reqdata)
            self.jobid = hosttabledict['jobid']
            self.name = hosttabledict['name']
            self.start_time = hosttabledict['start_time']
            self.hosttable = hosttabledict['hosttable']
            self.hostcolum = hosttabledict['hostcolum']
        return 0, "success"

    def func_hostcsvinfo(self, trancode, uuid, reqdata):
        if trancode == "hostcsvinfo":
            hosttabledict = reqdata
            print("hostcsvdict=", reqdata)
            self.jobid = hosttabledict['jobid']
            self.name = hosttabledict['name']
            self.start_time = hosttabledict['start_time']
            self.hostcsv = hosttabledict['hostcsv']
            self.hostcolum = hosttabledict['hostcolum']
        return 0, "success"

# if __name__ == '__main__':
#     SecureInformationRetrievalHost().key_derivation(1, 2, 3)
