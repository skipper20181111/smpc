import os

import numpy as np
import pandas as pd
import time
from collections import Counter
import random
import pickle

from baseCommon.baseConvert import getDirectValue
from baseCommon.conf_yaml import load_yaml_conf
from baseCommon.projectConf import get_project_base_directory
from baseCommon.httpcall import Httpapi
import numpy as np
import random
import os
import pickle

'''
G端：任务发起方，拥有x值和y值
H端：联合训练方，拥有x值
'''

from baseCommon.httpcall import Httpapi

'''
调用java端接口，根据dataName得到filePath
'''


def pickle_save(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()


def pickle_load(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


'''zmh_2021/9/10'''


def getpath_by_jobid(jobid, method):
    path = r"/data/zhanghui/woe/" + jobid + method + '.txt'
    return path


def get_path_by_name(name):
    http = Httpapi()
    response = http.http_post(url='/smpc/api/dataInfo/getFilePathByDataName', body={"dataName": name})
    path = response["result"]["filePath"]
    return path


'''
计算IV值的data进行文件名转化，转化为原始文件名
@param dataname
@return filename
'''


def inter_dataname_TANS(dataname):
    filename = dataname[:-10]
    return filename


'''
训练的data进行文件名转化，转化为原始文件名
@param dataname
@return filename
'''


def woe_dataname_TANS(dataname):
    filename = dataname[:-14]
    return filename


def model_save_upgrade(parameter, jobid, method):
    save_path = getpath_by_jobid(jobid, method)
    pickle_save(save_path, parameter)


'''
拼路径
'''


def concat_path(jobid, filepath):
    job_path = os.path.join(os.path.dirname(os.path.dirname(filepath)), jobid)
    return job_path


'''
拼文件地址的绝对路径
@param dataname
@param jobid
@param filepath
@return job_filepath
'''


def concat_filepath(dataname, jobid, filepath):
    job_filepath = os.path.join(os.path.dirname(os.path.dirname(filepath)), jobid, dataname)
    return job_filepath


def save_controll_data(controll_data, jobid, method):
    save_path = getpath_by_jobid(jobid, method)
    pickle_save(save_path, controll_data)


'''
将字符串格式的列表编程真正的列表
'''
def st2list(columns):
    columns = columns[1:-1]
    columns = columns.replace('\'', '')
    columns = columns.replace(' ', '')
    columns = columns.split(',')
    return columns




'''
将预测所得y返回到前端
'''


def http_feed_back(result):
    http = Httpapi()
    response = http.http_post(url='/smpc/api/dataInfo/getFilePathByDataName', body={"dataName": result})
    path = response["result"]["filePath"]


'''
由前端所给文件名找到在数据库中的路径
'''


# def get_file_path_by_http(filename):
#     http = Httpapi()
#     response = http.http_post(url='/smpc/api/dataInfo/getFilePathByDataName', body={"dataName": filename})
#     path = response["result"]["filePath"]
#     return path

# if __name__ == '__main__':
#     path=get_file_path_by_http(filename='1111.txt')
#     print(path)
def pickle_save(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file)
    file.close()


def pickle_load(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


'''zmh_2021/9/10'''


def getpath_by_jobid(jobid, method):
    path = r"/data/zhanghui/woe/" + jobid + method + '.txt'
    return path


# def getpath_by_jobid(path,jobid,method):
#     filename = jobid+method+'.txt'
#     path = os.path.join(os.path.dirname(os.path.dirname(path)),jobid,filename)
#     return path


def get_small(a, b):
    if a < b:
        return a, False
    else:
        return b, True


def formatstr(number):
    return str('%.20f' % number)


def pinstr(save_woe, feacher):
    cut_point_list = save_woe[feacher][1]
    str_list = []
    for i in range(len(cut_point_list)):
        if i == 0:
            str_list.append('[-inf,' + formatstr(cut_point_list[i]) + ')')
        elif i == (len(cut_point_list) - 1):
            str_list.append('[' + formatstr(cut_point_list[i - 1]) + ',' + formatstr(cut_point_list[i]) + ')')
            str_list.append('[' + formatstr(cut_point_list[i]) + ',inf]')
        else:
            str_list.append('[' + formatstr(cut_point_list[i - 1]) + ',' + formatstr(cut_point_list[i]) + ')')
    return str_list


def get_res_continu(save_woe, feacher):
    bin_list = pinstr(save_woe, feacher)
    woe_list = save_woe[feacher][2][1]
    result_list = []
    for i in range(len(bin_list)):
        result_list.append([bin_list[i], formatstr(woe_list[i])])
    return result_list


def get_res_descret(save_woe, feacher):
    bin_list = pinstrdes(save_woe, feacher)
    woe_list = save_woe[feacher][2][1]
    result_list = []
    for i in range(len(bin_list)):
        result_list.append([bin_list[i], formatstr(woe_list[i])])
    return result_list


def pinstrdes(save_woe, feacher):
    diction = save_woe[feacher][1][1]
    str_list = []
    for dick in diction:
        str_list.append(str(diction[dick]))
    return str_list


def get_woe_result(save_woe, feacher):
    if save_woe[feacher][0] == 1:
        return get_res_continu(save_woe, feacher)
    if save_woe[feacher][0] == 2:
        return get_res_descret(save_woe, feacher)


def find_bin(bin_threshold, sample_number):
    count = 0
    bin_num = 0
    for bin_threshold_point in bin_threshold:
        count += 1
        if sample_number >= bin_threshold_point:
            bin_num = count
    return bin_num


def sort_colu(data, feacher):
    return data.sort_values(by=[feacher])


def sort_colu_give(data, feacher):
    return data.sort_values(by=[feacher]).index


def sort_colu_get(data, inde):
    return data.iloc[inde]


def give_cut_point_mean_num(m, bin_number, ylist, feacherlist):
    y_list = []
    feacher_list = []
    cut_len = int(m / bin_number)
    for i in range(bin_number):
        if i <= bin_number - 2:
            listed_y = list(ylist[i * cut_len:(i + 1) * cut_len])
            y_list.append(listed_y)
            feacher_list.append(feacherlist[i * cut_len])
        else:
            y_list.append(list(ylist[i * cut_len:]))
            feacher_list.append(feacherlist[i * cut_len])
    return y_list, feacher_list


def woeiv(father_category, father_category_count, children_category_count):
    good_code = father_category[0]
    bad_code = father_category[1]
    badi = children_category_count[bad_code]
    badt = father_category_count[bad_code]
    goodi = children_category_count[good_code]
    goodt = father_category_count[good_code]
    if badi == 0 or goodi == 0:
        woe = np.log(((badi + 0.5) / (goodi + 0.5)) / (badt / goodt))
    else:
        woe = np.log(badi / badt) - np.log(goodi / goodt)
    iv = ((badi / badt) - (goodi / goodt)) * woe
    return woe, iv


def insert_encrypt_y(data, y):
    datafram = data.copy()
    datafram.insert(1, 'encrypt_y', y)
    return datafram


def insert_mask_y(data, y):
    datafram = data.copy()
    datafram.insert(1, 'y', y)
    return datafram


def give_bin_ypoint_and_ylist(method, data, feacher, bin_number):
    if method == 'mean_num_continues':
        return bin_mean_num(data, feacher, bin_number)
    elif method == 'mean_num_descrete':
        return bin_mean_num_descrete(data, feacher, bin_number)


def bin_mean_num_descrete(data, feacher, bin_number):
    data_for_calculate = data.copy()
    data_for_calculate = sort_colu(data_for_calculate, feacher)

    feacher_category = list(set(data_for_calculate[feacher]))
    if len(feacher_category) >= 2 * bin_number:
        m, n = data_for_calculate.shape
        y_list, feacher_list = give_cut_point_mean_num(m, bin_number, data_for_calculate.iloc[:, 1].values,
                                                       data_for_calculate[feacher].values)
        return feacher_list, y_list
    else:
        m, n = data_for_calculate.shape
        y_list = discrete_ylist(data, feacher, feacher_category)
        return feacher_category, y_list


def bin_mean_num(data, feacher, bin_number):
    data_for_calculate = data.copy()
    data_for_calculate = sort_colu(data_for_calculate, feacher)
    m, n = data_for_calculate.shape
    y_list, feacher_list = give_cut_point_mean_num(m, bin_number, data_for_calculate.iloc[:, 1].values,
                                                   data_for_calculate[feacher].values)
    return feacher_list, y_list  #### < 和>= 是分割点，这个很重要，不要忘记了


def bin_mean_distence(data, feacher, bin_number):
    data_for_calculate = data.copy()
    data_for_calculate = sort_colu(data_for_calculate, feacher)


def calculate_feacher_woeiv(data, y_list):
    category = list(set(data.iloc[:, 1]))
    father_category_count = Counter(data.iloc[:, 1])
    woelist = []
    ivlist = []
    for binylist in y_list:
        children_category_count = Counter(binylist)
        woe, iv = woeiv(category, father_category_count, children_category_count)
        woelist.append(woe)
        ivlist.append(iv)
    return woelist, sum(ivlist)


def calculate_ca_square(category, a_category_count, b_category_count):
    good_code = category[0]
    bad_code = category[1]
    bada = a_category_count[bad_code]
    gooda = a_category_count[good_code]
    badb = b_category_count[bad_code]
    goodb = b_category_count[good_code]
    a = bada
    b = gooda
    c = badb
    d = goodb
    n = a + b + c + d
    if (a + b) == 0 or (c + d) == 0:
        return 0
    if (a + c) == 0 or (b + d) == 0:
        return 10000
    ca_square = (n * (a * d - b * c) ** 2) / ((a + b) * (a + c) * (c + d) * (b + d))
    return ca_square


def to_file(path, obj):
    f = open(path, 'wb')
    pickle.dump(obj, f, 2)
    f.close()


def read_file(path):
    f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def rangewo(num):
    k = []
    for i in range(num):
        k.append(i)
    return k


def calculate_ca_square_list(category, y_list):
    ca_square_list = []
    for i in range(len(y_list) - 1):
        ca_square_list.append(calculate_ca_square(category, Counter(y_list[i]), Counter(y_list[i + 1])))
    return ca_square_list


def del_min_ca_square(y_list, cutlist, ca_threshold, bin_num_threshold, category):
    ca_square_list = calculate_ca_square_list(category, y_list)
    min_ca2 = min(ca_square_list)
    if min_ca2 <= ca_threshold or len(y_list) > bin_num_threshold:  ### 这个地方应当好好的问问言零零
        delet_index = ca_square_list.index(min_ca2)
        y_list = list_extend(y_list, [delet_index, delet_index + 1])
        del cutlist[delet_index + 1]
        return True, cutlist, y_list
    else:
        return False, cutlist[1:], y_list


def get_cut_point_number_list(cutlist, feacher_list):
    cut_number_list = []
    for i in cutlist:
        cut_number_list.append(feacher_list[i])
    return cut_number_list  ##### 这个值一定要返回给前端，不然唐老师该生气了


def continuous_encode(data, cut_number_list, feacher):
    m, n = data.shape
    for i in range(m):
        residual = list(abs(np.array(cut_number_list) - data.loc[i, feacher]))
        blockindex = residual.index(min(residual))
        if data.loc[i, feacher] < residual[blockindex]:
            data.loc[i, feacher] = blockindex
        else:
            data.loc[i, feacher] = blockindex + 1
    return data


def woe_transform(feacher_category, woelist, data, feacher):
    m, n = data.shape
    for i in range(m):
        a=int(data.loc[i, feacher])
        data.loc[i, feacher] = woelist[feacher_category.index(a)]
    return data


def encryptfun(y):
    return y


def discrete_encode(data, feacher):
    feacher_category = list(set(data[feacher]))
    random.shuffle(feacher_category)
    m, n = data.shape
    for i in range(m):
        data.loc[i, feacher] = int(feacher_category.index(data.loc[i, feacher]))
    return data, feacher_category


def discrete_ylist(data, feacher, feacher_category):
    y_list = []
    for i in feacher_category:
        y_list.append(list(data[data[feacher] == i].iloc[:, 1]))
    return y_list


def get_mini_discrete_combination(category, y_list, feacher_list):
    inum = len(y_list)
    min_ca_square = 1000  ##一般情况下，不可能有这么大的卡方值
    for i in range(inum):
        for j in range(inum - i - 1):
            min_ca_square, flag = get_small(min_ca_square, calculate_ca_square(category, Counter(y_list[i]),
                                                                               Counter(y_list[j + i + 1])))
            if flag == True:
                combination = [int(feacher_list[i]), int(feacher_list[j + i + 1])]
    return combination, min_ca_square


def discrete_combination(data, feacher, combination):
    m, n = data.shape
    for i in range(m):
        ppap = data.loc[i, feacher]
        if ppap == combination[0] or ppap == combination[1]:
            data.loc[i, feacher] = combination[0]
    return data


def list_extend(y_list_dup, combination_index):
    y_list_dup[combination_index[0]].extend(y_list_dup[combination_index[1]])
    del y_list_dup[combination_index[1]]
    return y_list_dup


def feacher_and_y_list_extend(y_list, feacher_list, combination_index):
    index1 = feacher_list.index(combination_index[0])
    index2 = feacher_list.index(combination_index[1])
    y_list = list_extend(y_list, [index1, index2])
    del feacher_list[index2]
    return y_list, feacher_list


def combination_dictionary(combination_list, feacher_category, feacher_category_index):
    category_dict = {}
    for each_feacher_category_index in feacher_category_index:
        each_combination_list = get_combination_list(combination_list, each_feacher_category_index)
        each_feacher_category_list = []
        for num in each_combination_list:
            each_feacher_category_list.append(feacher_category[num])
        category_dict[each_feacher_category_index] = each_feacher_category_list
    return category_dict


def get_combination_list(combination_list, feacher_category):
    combination_lists = combination_list.copy()
    same_category = [feacher_category]
    have = 0
    while True:
        have = 0
        for comb in combination_lists:
            if (comb[0] in same_category) or (comb[1] in same_category):
                have += 1
                same_category.append(comb[0])
                same_category.append(comb[1])
                combination_lists.remove(comb)
        if have == 0:
            break
    return list(set(same_category))


def continuation_combination(feacher, data, feacher_cut_point_list):
    m, n = data.shape
    for i in range(m):
        ppap = data.loc[i, feacher]
        diverse = list(map(lambda x: abs(x - ppap), feacher_cut_point_list))
        index = diverse.index(min(diverse))
        if ppap < feacher_cut_point_list[index]:
            data.loc[i, feacher] = index -1
        else:
            data.loc[i, feacher] = index
    return data


def getlinearModePathPara(para=None):
    if para is None:
        loadyaml = load_yaml_conf("gobal_conf.yaml")
    else:
        loadyaml = load_yaml_conf(para)
    return getDirectValue(loadyaml, 'woe_detail')


def get_path():
    datapath = getlinearModePathPara("woe_iv_ca_Model.yaml")
    project = os.getenv("PROJECTPATH")
    if project is None:
        project = get_project_base_directory()

    datapath = project + "/{}".format(datapath)

    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath
