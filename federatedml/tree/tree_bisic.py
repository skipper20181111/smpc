import numpy as np
import pandas as pd
from collections import Counter

'''
这是一个小小的树,我种下一颗种子，终于长出了果实，今天是个伟大日子。

'''


class hyper:
    array = 10
    bin_number = 400

    ### 这个就是可以简单实现的四种之简单的两种


def tree_gini(p):
    return p * (1 - p)


def tree_bit(p):
    if p == 0:
        return p
    return abs(-p * np.log2(p))

    ### 这个就是排序的算法，已经是python下的最优结果了


def sort_colu(x, i):
    b = x[:, i]
    index = np.lexsort((b,))
    return x[index]
    ### 排序的结果有时需要传递给另一端，因此需要新的排序算法


def sort_colu_give(x, i):
    return np.lexsort((x[:, i],))


def sort_colu_get(x, inde):
    return x[inde]

    ### 最小值算法，这个总没有机会优化吧


def get_small(a, b):
    if a < b:
        return a, False
    else:
        return b, True

    ###  根据排序结果进行矩阵的gini计算


def get_array_gini(x, category):
    gini = 0
    cat = Counter(x[:, -1])
    m = np.shape(x)[0]
    for i in category:
        gini += tree_gini(cat[i] / m)
    return gini


def get_mini_gini(xy):
    category = set(xy[:, -1])
    m, n = np.shape(xy)
    min_gini = 100
    bin_num, if_bin = get_small(hyper.bin_number, m)
    #####在样本数量小于最大值和大于最大值两种情况下，要分别对待。
    if if_bin:

        for i in range(1, m - 1):
            a = ((i / m) * get_array_gini(xy[:i, :], category))
            b = (((m - i) / m) * get_array_gini(xy[i:, :], category))
            min_gini, flag = get_small(min_gini, (a + b))
            if flag:
                cut_point = i
    else:
        binPartision = int(m / hyper.bin_number)
        for i in range(1, int(m / binPartision) - 1):

            a = ((i * binPartision / m) * get_array_gini(xy[:i * binPartision, :], category))
            b = (((m - i * binPartision) / m) * get_array_gini(xy[i * binPartision:, :], category))
            min_gini, flag = get_small(min_gini, (a + b))
            if flag:
                cut_point = i * binPartision

    return min_gini, cut_point


def china(index, cut_point, xy):
    m, n = np.shape(xy)
    if cut_point == m or cut_point == 0:
        return xy, xy, False
    x = xy.copy()
    x = sort_colu_get(x, index)
    cut_left = x[:cut_point, :]
    cut_right = x[cut_point:, :]
    return cut_left, cut_right, True


def last_china_B(xy):
    m, n = np.shape(xy)
    x = xy.copy()
    x = sort_colu(x, 0)
    mini_gini, cut_point = get_mini_gini(x)

    return [('b', 0, x[cut_point, 0]), [(-1, -1, round(sum(x[:cut_point, -1]) / (cut_point))), [], []],
            [(-1, -1, round(sum(x[cut_point:, -1]) / (m - cut_point))), [], []]]


def last_china_A(xy):
    x = xy.copy()
    inde = sort_colu_give(x, 0)
    x = sort_colu_get(x, inde)
    mini_gini, cut_point = get_mini_gini(x)
    return x[cut_point, 0], inde, cut_point


def last_china_A_B(xy, inde, cut_point):
    m, n = np.shape(xy)
    x = xy.copy()
    x = sort_colu_get(x, inde)
    up_count = round(sum(x[:cut_point, -1]) / (cut_point))
    down_count = round(sum(x[cut_point:, -1]) / (m - cut_point))
    return up_count, down_count
