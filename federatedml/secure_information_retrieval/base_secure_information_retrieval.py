import functools
from baseCommon.projectConf import get_project_base_directory, conf_realpath


class BaseSecureInformationRetrieval(object):
    def __init__(self):
        self.security_level = None
        self.commutative_cipher = None
        self.transfer_variable = None
        self.block_num = None
        self.coverage = None


class CryptoExecutor(object):
    def __init__(self, cipher_core):
        self.cipher_core = cipher_core

    def init(self):
        self.cipher_core.init()

    def reinit(self, cipher_core):
        self.cipher_core = cipher_core

    def map_encrypt(self, plaintable, mode):
        if mode == 0:
            return list(map(lambda x: [x[0], self.cipher_core.encrypt(x[0])], plaintable))
            # return plaintable.map(lambda k, v: (k, self.cipher_core.encrypt(k)))
        elif mode == 1:
            return list(map(lambda x: [self.cipher_core.encrypt(x[0]), -1], plaintable))
            # return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), -1))
        elif mode == 2:
            return list(map(lambda x: [self.cipher_core.encrypt(x[0]), x[1:]], plaintable))
            # return plaintable.map(lambda k, v: (self.cipher_core.encrypt(k), v))
        elif mode == 3:
            return list(map(lambda x: [x[0], (self.cipher_core.encrypt(x[0]), x[1:])], plaintable))
            # return plaintable.map(lambda k, v: (k, (self.cipher_core.encrypt(k), v)))
        else:
            raise ValueError("Unsupported mode for crypto_executor map encryption")

    def map_decrypt(self, ciphertable, mode):
        if mode == 0:
            return list(map(lambda x: [x[0], [self.cipher_core.decrypt(i) for i in x[1:]]], ciphertable))
            # return ciphertable.map(lambda k, v: (k, self.cipher_core.decrypt(k)))
        elif mode == 1:
            return list(map(lambda x: [self.cipher_core.decrypt(x[0]), -1], ciphertable))
            # return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), -1))
        elif mode == 2:
            return list(map(lambda x: [self.cipher_core.decrypt(x[0]), x[1:]], ciphertable))
            # return ciphertable.map(lambda k, v: (self.cipher_core.decrypt(k), v))
        elif mode == 3:
            return list(map(lambda x: [x[0], (self.cipher_core.decrypt(x[0]), x[1:])], ciphertable))
            # return ciphertable.map(lambda k, v: (k, (self.cipher_core.decrypt(k), v)))
        else:
            raise ValueError("Unsupported mode for crypto_executor map decryption")

    def map_values_decrypt(self, ciphertable, mode):
        if mode == 0:
            return ciphertable.mapValues(lambda v: self.cipher_core.decrypt(v))
        elif mode == 1:
            f = functools.partial(self.cipher_core.decrypt, decode_output=True)
            return ciphertable.mapValues(lambda v: f(v))
        else:
            raise ValueError("Unsupported mode for crypto_executor map_values encryption")

    def get_nonce(self):
        return self.cipher_core.get_nonce()
