import pandas as pd
import time
from collections import Counter
import numpy as np
from federatedml.tree.tree_bisic import *
import random
import pickle
from secureprotol.encrypt import PaillierEncrypt

from baseInterface import utilsApi, model_pb2_grpc, modelClientApi

dataload = np.load(r'f:\tree.npz')
xy_A = dataload['arr_0']
xy_B = dataload['arr_1']
xy = np.hstack((xy_A, xy_B))
for i in range(6):
    xy = np.vstack((xy, xy))
print(np.shape(xy))
# xy=xy[:,(4,0,1,2,3)]
xy_A = xy[:, (0, 1)]
xy_B = xy[:, 2:]
m, n = np.shape(xy)
xyd_A = pd.DataFrame(xy_A)
xyd_B = pd.DataFrame(xy_B)


# tree_bisic.get_array_gini(xy,set(xy[:,-1]))

class tree_single():
    def __init__(self):
        super().__init__()
        # self.uuid = None
        self.public_key = None
        self.privacy_key = None
        self.stub = None
        self.grpcclient = None
        self.data = None
        self.alldata = None
        self.testdata = None
        self.w = None

    def test(self):
        xy_A_feacher = xyd_A.columns
        xy_A_sample = xyd_A.index
        xy_A_info = []
        xy_A_info.append(xy_A_sample)
        xy_A_info.append(xy_A_feacher)

        self.grpcclient = modelClientApi.GrpcClient("appconf.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        ''' 第一次传输xy_A的信息 有na，还有列与行标记位'''

        req = self.grpcclient.request_from_OnetoOne(trancode='cutpoint_gini_list', uuid='uuid_cutpoint_gini_list',
                                                    reqdata=xy_A_info)
        response = self.stub.OnetoOne(req)
        A_response = pickle.loads(response.respdata)
        na = A_response[0]  # 返回xy_A的信息
        xy_b = xy_B[:, :].copy()
        m, n = np.shape(xy_b)
        category = set(xy_B[:, -1])
        children_gini = get_array_gini(xy_B, category)
        gini_list = []
        cut_point_list = []
        for i in range(n - 1):
            xy_b = sort_colu(xy_b, i)
            mini_gini, cut_point = get_mini_gini(xy_b)
            gini_list.append(mini_gini)
            cut_point_list.append(cut_point)
        print('yes')
        for i in range(na):
            xy_b = sort_colu_get(xy_B, A_response[i + 1])
            mini_gini, cut_point = get_mini_gini(xy_b)
            gini_list.append(mini_gini)
            cut_point_list.append(cut_point)
        print(gini_list, cut_point_list)
        '''极其重要的极其复杂的一步， 快把我的头发调掉了。'''
        feacher_num = gini_list.index(min(gini_list))
        print(feacher_num)
        if feacher_num <= (n - 2):
            index_sort = sort_colu_give(xy_B, feacher_num)
            cut_point = cut_point_list[feacher_num]
            ''' china_pramerter 生成之后，传到grpc另一端，另一端调用正常的china函数来生成结果信息，再传回 B 端'''

            china_parameter = [index_sort, cut_point, xy_A_info]
            req = self.grpcclient.request_from_OnetoOne(trancode='grpc_china_ifB', uuid='uuid_grpc_china_ifB',
                                                        reqdata=china_parameter)
            response = self.stub.OnetoOne(req)
            china_response = pickle.loads(response.respdata)
            xal, xar, suca = china_response[0], china_response[1], china_response[2]
            xbl, xbr, sucb = china(index_sort, cut_point, xy_B)

            xbr = np.hstack((xbr[:, :feacher_num], xbr[:, feacher_num + 1:]))
            xbl = np.hstack((xbl[:, :feacher_num], xbl[:, feacher_num + 1:]))
            cut_point_num = sort_colu_get(xy_B, index_sort)[cut_point, feacher_num]


        else:
            cut_point = cut_point_list[feacher_num]
            feacher_num = feacher_num - n + 1
            '''这里明显又要重写，  由于是在 A 端进行，因此需要传过去feacher_num 和 cut_point ,xy_A_info
            然后回传 xal，xar，index_sort，cutpoint_num 
            注意，这里的xal和xar已经删掉了部分feacher，因此不需要再进行处理。
            '''
            china_parameter = [feacher_num, cut_point, xy_A_info]
            req = self.grpcclient.request_from_OnetoOne(trancode='grpc_china_ifA', uuid='uuid_grpc_china_ifA',
                                                        reqdata=china_parameter)
            response = self.stub.OnetoOne(req)
            china_response = pickle.loads(response.respdata)
            xal, xar, suca = china_response[0], china_response[1], china_response[-1]
            index_sort = china_response[2]
            cut_point_num = china_response[3]
            xbl, xbr, sucb = china(index_sort, cut_point, xy_B)

    # xy_B是拥有y值的那一端
    def recurrence(self, xy_A_info, xy_B, father_gini):
        self.grpcclient = modelClientApi.GrpcClient("appconf.yaml")  # 初始化
        self.stub = self.grpcclient.build(connectpara=None, secureflag=0)  # 建立通道
        ''' 第一次传输xy_A的信息 有na，还有列与行标记位'''

        req = self.grpcclient.request_from_OnetoOne(trancode='cutpoint_gini_list', uuid='uuid_cutpoint_gini_list',
                                                    reqdata=xy_A_info)
        response = self.stub.OnetoOne(req)
        A_response = pickle.loads(response.respdata)
        na = A_response[0]  # 返回xy_A的信息
        xy_b = xy_B[:, :].copy()
        m, n = np.shape(xy_b)
        category = set(xy_B[:, -1])

        children_gini = get_array_gini(xy_B, category)
        # 如果切分后的类过于单一，那么直接返回叶子节点即可
        if children_gini <= 0.1:
            return [(-1, -1, round(sum(xy_B[:, -1]) / m)), [], []]

        gini_list = []
        cut_point_list = []
        for i in range(n - 1):
            xy_b = sort_colu(xy_b, i)
            mini_gini, cut_point = get_mini_gini(xy_b)
            gini_list.append(mini_gini)
            cut_point_list.append(cut_point)
        print('yes')
        for i in range(na):
            xy_b = sort_colu_get(xy_B, A_response[i + 1])
            mini_gini, cut_point = get_mini_gini(xy_b)
            gini_list.append(mini_gini)
            cut_point_list.append(cut_point)
        print(gini_list, cut_point_list)

        # 如果切分后的类过于单一，那么直接返回叶子节点即可
        if children_gini <= 0.1:
            return [(-1, -1, round(sum(xy_B[:, -1]) / m)), [], []]

        # 如果只剩了一个特征，那么直接计算叶子节点即可，但是，这个先不写。
        if (n + na) == 2:
            if n == 1:
                ## B方啥也不剩，A方还剩一个
                ''' B方啥也不剩，A方还剩一个的情况下  生成last_china_pramerter ，传到grpc另一端，另一端调用正常的china函数来生成结果信息，再传回 B 端'''

                china_parameter = [xy_A_info]
                req = self.grpcclient.request_from_OnetoOne(trancode='last_china', uuid='uuid_last_china',
                                                            reqdata=china_parameter)
                response = self.stub.OnetoOne(req)
                china_response = pickle.loads(response.respdata)
                cut_num, inde, cut_point = china_response
                up_count, down_count = last_china_A_B(xy_B, inde, cut_point)
                if up_count == down_count:
                    return [(-1, -1, up_count), [], []]
                else:
                    return [('a', 0, cut_num), [(-1, -1, up_count), [], []], [(-1, -1, down_count), [], []]]
            else:
                ## A方啥也不剩，B方还剩一个

                return last_china_B(xy_B)

        # 如果矩阵的样本数量小于阈值，那么直接返回叶子节点即可：
        if m <= 5:
            return [(-1, -1, round(sum(xy_B[:, -1]) / m)), [], []]

        # 如果gini变化太小，则直接返回叶子节点
        if (min(gini_list) / father_gini) > 0.9:
            return [(-1, -1, round(sum(xy_B[:, -1]) / m)), [], []]

        '''极其重要的极其复杂的一步，快把我的头发调掉了。'''
        feacher_num = gini_list.index(min(gini_list))
        print(feacher_num)
        if feacher_num <= (n - 2):
            index_sort = sort_colu_give(xy_B, feacher_num)
            cut_point = cut_point_list[feacher_num]
            ''' china_pramerter 生成之后，传到grpc另一端，另一端调用正常的china函数来生成结果信息，再传回 B 端'''

            china_parameter = [index_sort, cut_point, xy_A_info]
            req = self.grpcclient.request_from_OnetoOne(trancode='grpc_china_ifB', uuid='uuid_grpc_china_ifB',
                                                        reqdata=china_parameter)
            response = self.stub.OnetoOne(req)
            china_response = pickle.loads(response.respdata)
            xal, xar, suca = china_response[0], china_response[1], china_response[2]
            xbl, xbr, sucb = china(index_sort, cut_point, xy_B)

            xbr = np.hstack((xbr[:, :feacher_num], xbr[:, feacher_num + 1:]))
            xbl = np.hstack((xbl[:, :feacher_num], xbl[:, feacher_num + 1:]))
            cut_point_num = sort_colu_get(xy_B, index_sort)[cut_point, feacher_num]
            if suca == False or sucb == False:  # 矩阵没有切成功，直接返回叶子节点
                return [(-1, -1, round(sum(xy_B[:, -1]) / m)), [], []]
            else:
                return [('b', feacher_num, cut_point_num), self.recurrence(xal, xbl, children_gini),
                        self.recurrence(xar, xbr, children_gini)]


        else:
            cut_point = cut_point_list[feacher_num]
            feacher_num = feacher_num - n + 1
            '''这里明显又要重写，  由于是在 A 端进行，因此需要传过去feacher_num 和 cut_point ,xy_A_info
            然后回传 xal，xar，index_sort，cutpoint_num 
            注意，这里的xal和xar已经删掉了部分feacher，因此不需要再进行处理。
            '''
            china_parameter = [feacher_num, cut_point, xy_A_info]
            req = self.grpcclient.request_from_OnetoOne(trancode='grpc_china_ifA', uuid='uuid_grpc_china_ifA',
                                                        reqdata=china_parameter)
            response = self.stub.OnetoOne(req)
            china_response = pickle.loads(response.respdata)
            xal, xar, suca = china_response[0], china_response[1], china_response[-1]
            index_sort = china_response[2]
            cut_point_num = china_response[3]
            xbl, xbr, sucb = china(index_sort, cut_point, xy_B)
            if suca == False or sucb == False:  # 矩阵没有切成功，直接返回叶子节点
                return [(-1, -1, round(sum(xy[:, -1]) / m)), [], []]
            else:
                return [('a', feacher_num, cut_point_num), self.recurrence(xal, xbl, children_gini),
                        self.recurrence(xar, xbr, children_gini)]

    def forecast_tree(self, tree_structure, sample_a, sample_b):
        gofeacher = tree_structure[0]
        if gofeacher[0] == -1:
            return gofeacher[2]

        if gofeacher[0] == 'b':
            if sample_b[gofeacher[1]] >= gofeacher[2]:
                sample_b = np.hstack((sample_b[:gofeacher[1]], sample_b[(gofeacher[1] + 1):]))
                ret = self.forecast_tree(tree_structure[2], sample_a, sample_b)
            else:
                sample_b = np.hstack((sample_b[:gofeacher[1]], sample_b[(gofeacher[1] + 1):]))
                ret = self.forecast_tree(tree_structure[1], sample_a, sample_b)
        else:  ### 这个一定是a
            if sample_a[gofeacher[1]] >= gofeacher[2]:
                sample_a = np.hstack((sample_a[:gofeacher[1]], sample_a[(gofeacher[1] + 1):]))
                ret = self.forecast_tree(tree_structure[2], sample_a, sample_b)
            else:
                sample_a = np.hstack((sample_a[:gofeacher[1]], sample_a[(gofeacher[1] + 1):]))
                ret = self.forecast_tree(tree_structure[1], sample_a, sample_b)
        return ret


if __name__ == '__main__':
    xy_A_feacher = xyd_A.columns
    xy_A_sample = xyd_A.index
    xy_A_info = []
    xy_A_info.append(xy_A_sample)
    xy_A_info.append(xy_A_feacher)
    lh = tree_single()
    tree_structure = lh.recurrence(xy_A_info, xy_B, -1)
    print(tree_structure)
    count_true = 0
    for i in range(np.shape(xy)[0]):

        sa = xy_A[i, :]
        sb = xy_B[i, :]
        print(lh.forecast_tree(tree_structure, sa, sb) == sb[-1])
        if lh.forecast_tree(tree_structure, sa, sb) == sb[-1]:
            count_true = count_true + 1
    print(count_true / np.shape(xy)[0])
