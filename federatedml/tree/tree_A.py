import pandas as pd
import time
from collections import Counter
import numpy as np
from federatedml.tree.tree_bisic import *
from baseCommon.plogger import *

LoggerFactory.set_directory(directory="./")
LOGGER = getLogger()

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
xyd_A = pd.DataFrame(xy_A)
xyd_B = pd.DataFrame(xy_B)


class tree_A():

    def grpc_china_ifA(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "grpc_china_ifA":
            feacher_num = reqdata[0]
            cut_point = reqdata[1]
            xy_A_info = reqdata[2]
            xyd_a = xyd_A.copy()
            xy_A_feacher = xy_A_info[1]
            xy_A_sample = xy_A_info[0]
            xyd_a = xyd_a.loc[xy_A_sample, xy_A_feacher]
            xy_a = np.array(xyd_a)
            '''因为要重写china，所以就在这里重写一下吧'''
            m, n = xyd_a.shape
            index_sort = sort_colu_give(xy_a, feacher_num)
            if cut_point == m or cut_point == 0:
                response = [xy_A_info, xy_A_info, index_sort, -100, False]
            xyd_a = xyd_a.iloc[index_sort, :]

            cut_left_index = xyd_a.iloc[:cut_point, :].index
            cut_right_index = xyd_a.iloc[cut_point:, :].index
            xy_A_cut_feacher = xy_A_feacher.tolist()[:feacher_num] + xy_A_feacher.tolist()[feacher_num:]
            left_info = [cut_left_index, xy_A_cut_feacher]
            right_info = [cut_right_index, xy_A_cut_feacher]

            cut_point_num = sort_colu_get(xy_a, index_sort)[cut_point, feacher_num]
            response = [left_info, right_info, index_sort, cut_point_num, True]
            return 0, response
        else:
            LOGGER.info("grpc_china_ifA Error!")
            return 500, "grpc_china_ifA Error!"

    def grpc_china_ifB(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "grpc_china_ifB":
            index_sort = reqdata[0]
            cut_point = reqdata[1]
            xy_A_info = reqdata[2]
            xyd_a = xyd_A.copy()
            xy_A_feacher = xy_A_info[1]
            xy_A_sample = xy_A_info[0]
            xyd_a = xyd_a.loc[xy_A_sample, xy_A_feacher]
            '''因为要重写china，所以就在这里重写一下吧'''
            m, n = xyd_a.shape
            if cut_point == m or cut_point == 0:
                response = [xy_A_info, xy_A_info, False]
            xyd_a = xyd_a.iloc[index_sort, :]

            cut_left_index = xyd_a.iloc[:cut_point, :].index
            cut_right_index = xyd_a.iloc[cut_point:, :].index
            left_info = [cut_left_index, xy_A_feacher]
            right_info = [cut_right_index, xy_A_feacher]
            response = [left_info, right_info, True]

            return 0, response
        else:
            LOGGER.info("grpc_china_ifB Error!")
            return 500, "grpc_china_ifB Error!"

    def last_china(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "last_china":
            xy_A_info = reqdata
            xyd_a = xyd_A.copy()
            xy_A_feacher = xy_A_info[1]
            xy_A_sample = xy_A_info[0]
            xyd_a = xyd_a.loc[xy_A_sample, xy_A_feacher]
            xy_a = np.array(xyd_a).reshape(xyd_a.shape[0], 1)
            indexs = sort_colu_give(xy_a, 0)
            xy_a = sort_colu_get(xy_a, indexs)
            mini_gini, cut_point = get_mini_gini(xy_a)
            response = [xy_a[cut_point, 0], indexs, cut_point]
            return 0, response
        else:
            LOGGER.info("last_china Error!")
            return 500, "last_china Error!"

    def cutpoint_gini_list(tradecode, uuid, reqdata):
        LOGGER.info(f"t={tradecode} uuid{uuid}  data={reqdata}")
        if tradecode == "cutpoint_gini_list":
            xyd_a = xyd_A.copy()
            print(reqdata)
            xy_A_feacher = reqdata[1]
            xy_A_sample = reqdata[0]
            xyd_a = xyd_a.loc[xy_A_sample, xy_A_feacher]
            xy_a = np.array(xyd_a)
            m, n = np.shape(xy_a)
            response = [n]
            for i in range(n):
                response.append(sort_colu_give(xy_a, i).tolist())
            return 0, response
        else:
            LOGGER.info("cutpoint_gini_list Error!")
            return 500, "cutpoint_gini_list Error!"

# tree_bisic.get_array_gini(xy,set(xy[:,-1]))
# for i in range(na):
#     xy_b = sort_colu_get(xy_B, sort_colu_give(xy_A, i))
#     mini_gini, cut_point = get_mini_gini(xy_b)
#     gini_list.append(mini_gini)
#     cut_point_list.append(cut_point)
