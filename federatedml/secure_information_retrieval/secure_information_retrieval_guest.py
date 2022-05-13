import random
from federatedml.secure_information_retrieval.base_secure_information_retrieval import BaseSecureInformationRetrieval, \
    CryptoExecutor
from secureprotol import gmpy_math
from secureprotol.pohlig_hellman_encryption import PohligHellmanCipherKey
from baseInterface import model_pb2_grpc, modelClientApi
import pickle
from computeApi.paralleCompute import *


class SecureInformationRetrievalGuest(BaseSecureInformationRetrieval):
    def __init__(self):
        super(SecureInformationRetrievalGuest, self).__init__()
        self.grpcclient = None
        self.stub = None
        self.uuid = 'SecureInformationRetrieval'
        self.block_num = 10
        self.r = None
        self.random_bit = 1024

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
        time_grpc = time.time()
        self.grpcclient = modelClientApi.GrpcClient()
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)
        print(f"time_grpc={time.time() - time_grpc}")

        time_key_transfer = time.time()
        # 1. 同步通讯秘钥共有知识
        self.sir_pk_transfer()
        print(f"time_key_transfer={time.time() - time_key_transfer}")
        print("key_transfer")

        # 2. 加密guest_id,得到Eg，发送，得到Eh
        time_E = time.time()
        id_list_guest_first = self._encrypt_id(guest_data, reserve_original_key=True)  # [g,Eg]
        id_list_guest_first_only = list(map(lambda x: [x[1], -1], id_list_guest_first))
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal1stid', uuid=self.uuid,
                                                                            reqdata=id_list_guest_first_only))
        id_list_host_first = pickle.loads(response.respdata)  # [Eh,-1]
        print(f"time_EH={time.time() - time_E}")
        print("Eh")

        # 3. 加密Eh，得到EEh，从host收到EEg
        time_EE = time.time()
        id_list_host_second = self._encrypt_id(id_list_host_first, reserve_original_key=True)  # [Eh,EEh]
        id_list_host_second_only = list(map(lambda x: [x[1], -1], id_list_host_second))
        response = self.stub.OnetoOne(self.grpcclient.request_from_OnetoOne(trancode='cal2ndid', uuid=self.uuid,
                                                                            reqdata="id_list_host_second_only"))
        id_list_guest_second = pickle.loads(response.respdata)  # [Eg,(EEg,-1)]
        print(f"time_EE={time.time() - time_EE}")
        print("EEh")

        time_reserve = time.time()

        id_list_guest_second_reserve = list(
            map(lambda x, y: [x[0], y[1][0]], id_list_guest_first, id_list_guest_second))

        print(f"time_reserve={time.time() - time_reserve}")

        id_list_guest_second_only = [(i[0], i[1]) for i in np.array(id_list_guest_second)[:, 1]]

        # 求交集
        time_intersect = time.time()
        id_list_intersect = self.find_intersection(id_list_guest_second_only, id_list_host_second_only)  # [EEi,-1]
        print(f"time_intersect={time.time() - time_intersect}")
        print("EEi")

        # 将intersect与guest id重建连接
        time_reserve_guest = time.time()
        id_list_intersect_reserve, keys = cmp(id_list_intersect, id_list_guest_second_reserve, 0, 1)
        print(f"time_reserve_guest={time.time() - time_reserve_guest}")
        time_fake = time.time()

        # 制造传输数据集，发送给host
        id_blocks = self.fake_blocks(id_list_intersect, id_list_host_second_only)  # List[(EEi, -1)]
        time_fake_transfer = time.time()
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="calreindexid", uuid=self.uuid, reqdata=id_blocks))  # 时间长
        print(f"time_fake_transfer={time.time() - time_fake_transfer}")
        print(f"time_fake={time_fake - time.time()}")

        # 发起OT操作，获得秘钥的list
        time_OT_key = time.time()
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="executeot", uuid=self.uuid, reqdata="Request OT Keys"))
        key_list = pickle.loads(response.respdata)
        print("OT")
        print(f"time_OT_key={time.time() - time_OT_key}")

        # 生成随机数，用目标秘钥进行加密，发送给host，获取密文数据
        time_random = time.time()
        self.r = random.randint(2 ** (1024 - 1), key_list[self.target_block_index][1] - 1)
        enc_r = gmpy_math.powmod(self.r, key_list[self.target_block_index][0], key_list[self.target_block_index][1])
        response = self.stub.OnetoOne(
            self.grpcclient.request_from_OnetoOne(trancode="getvalue", uuid=self.uuid, reqdata=enc_r))
        id_block_ciphertext = pickle.loads(response.respdata)
        print(f"time_random={time.time() - time_random}")

        # 将密文数据解密，获得查询结果
        final_time = time.time()
        target_blocks_value = []
        for i in id_block_ciphertext[self.target_block_index]:
            target_block_value = list(map(lambda x: self.cal_divm(x), i[1:]))
            target_blocks_value.append(target_block_value)
        target_blocks = np.hstack((np.array(id_list_intersect_reserve)[:, [0]], target_blocks_value)).tolist()
        print(f"target_blocks={target_blocks}")
        print(f"final_time={time.time() - final_time}")
        return target_blocks

    def cal_divm(self, value):
        value2 = gmpy_math.divm(value, self.r, 2 ** 1024)
        return int(value2)


if __name__ == '__main__':
    p = []
    for i in range(3):
        time_start = time.time()
        guestid = np.arange(10 ** (i + 2)).reshape(10 ** (i + 2), 1).tolist()
        SecureInformationRetrievalGuest().run(guestid)
        time_end = time.time()
        p.append(f"查询方数据量为={10 ** (i + 2)},total_time={time_end - time_start}秒")
    print(p)
