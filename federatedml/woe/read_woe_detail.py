import argparse
import ast

from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.extprint import ExPrint
from federatedml.woe.woe_iv_ca_square_basic import *

import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi
import multiprocessing as mp
import csv
import pandas as pd
import os


def get_controll_data(jobid, method):
    save_path = getpath_by_jobid(jobid, method)
    controll_data = pickle_load(save_path)
    return controll_data

def give_woe_detail(jobid, feacher, g_or_h):
    woe_controll = get_controll_data(jobid, 'woe')  # 读取woe相关数据
    save_woe_translate_guest, save_woe_translate_host = woe_controll
    if g_or_h == '1':
        detail = save_woe_translate_host
    else:
        detail = save_woe_translate_guest
    return get_woe_result(detail, feacher)

if __name__ == '__main__':
    a=give_woe_detail('1448199200478699522','x7','1')
    print(a)

