import hashlib
import random
import time
from multiprocessing import Pool

import numpy as np

from secureprotol import gmpy_math
from secureprotol.encrypt import RsaEncrypt

ek = 0
nk = 0
r = random.SystemRandom().getrandbits(128)


def GeneratePK():
    encrypt_operator = RsaEncrypt()
    encrypt_operator.generate_key()
    e, d, n = encrypt_operator.get_key_pair()
    public_key = {'e': e, 'n': n}
    return int(public_key['e']), int(public_key['n'])


def hash(value):
    return hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest()


'''
def Cal_G1(guest_id):
    g1 = gmpy_math.powmod(r,int(ek),int(nk)) * int(hash(guest_id),16) % int(nk)
    return g1
'''


def Cal_G1(guest_id):
    if isinstance(guest_id, list):
        guest_id = guest_id[0]
    g1 = gmpy_math.powmod(r, ek, nk) * int(hash(guest_id), 16) % nk
    '''
    g1=[]
    print("LEN=",guest_id)
    for  i in  range(len(guest_id)):
        gg=gmpy_math.powmod(r,int(ek),int(nk)) * int(hash(guest_id),16) % int(nk)
        g1.append(gg)
    '''
    return g1


def Get_G2(guest_id):
    # print('start')
    # with mp.Pool() as pool:
    tt1 = time.time()
    with Pool() as pool:
        # g1=list(pool.map(lambda x: [Cal_G1(x)],guest_id))
        print("LLL=", len(guest_id))
        g1 = list(pool.map(Cal_G1, guest_id))
    tt2 = time.time()
    # g2 = list(map(lambda x: [Cal_G1(x)],guest_id))
    g2 = []
    tt3 = time.time()
    print("tt2-tt1={} tt3-tt2={}".format(tt2 - tt1, tt3 - tt2))
    return g1, g2


def randint(start, end):
    return np.random.randint(start, end, [1, end - start + 1])


if __name__ == '__main__':
    print("HHHHHH")
    lis1 = list(randint(1, 2000000)[0])
    print("lis1=", lis1)
    ek, nk = GeneratePK()
    t1 = time.time()
    g1, g2 = Get_G2(lis1)
    t2 = time.time()
    print("tt=", t2 - t1)
    print("g1=", len(g1))
    print("g2=", len(g2))
