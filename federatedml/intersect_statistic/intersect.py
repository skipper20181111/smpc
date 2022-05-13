import hashlib


class Intersect(object):
    pass


class RsaIntersect(Intersect):
    def __init__(self):
        pass

    @staticmethod
    def hash(value):
        return hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest()

    def run(self, id):
        raise NotImplementedError("method init must be define")
