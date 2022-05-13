from federatedml.util import consts


class SecureInformationRetrievalParam(object):
    def __init__(self, security_level=0.5,
                 oblivious_transfer_protol=consts.OT_RSA,
                 commutative_encryption=consts.CE_PH,
                 key_size=1024,
                 raw_retrieval=False):
        self.security_level = security_level
        self.oblivious_transfer_protocol = oblivious_transfer_protocol
        self.commutative_encryption = commutative_encryption
        self.key_size = key_size
        self.raw_retrieval = raw_retrieval

    def check(self):
        descr = "secure information retrieval param's"
