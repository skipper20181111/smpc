import pandas as pd
import numpy as np


def stdnor(p):
    return (p - p.mean(axis=0)) / p.std(axis=0), [p.mean(axis=0), p.std(axis=0)]


''' flag是告诉函数要使用已有的正则化参数还是要生成正则化参数 parametersavepath  这个参数必须是.npy的格式，切记不要写错了
    datapath和datasavepath分别是数据来源和数据去向
    g_or_h是关键参数，有y的那一端是int(2)，没有y的那一端是int(1)
'''


def normalFeacher(flag, datapath, datasavepath, g_or_h, parametersavepath):
    saveparam = []
    if flag == True:  ##那么这个函数是要返回标准化的结果并存储标准化的结果
        data = pd.read_csv(datapath)
        for i in range(g_or_h, data.shape[1]):
            data.iloc[:, i], paramr = stdnor(data.iloc[:, i])
            saveparam.append(paramr)
        np.save(parametersavepath, saveparam)
        data.to_csv(datasavepath, index=0)
        return data
    else:  # 此时这个函数读取存储的标准化
        saveparam = np.load(parametersavepath)
        data = pd.read_csv(datapath)
        for i in range(g_or_h, data.shape[1]):
            data.iloc[:, i] = (data.iloc[:, i] - saveparam[i - g_or_h][0]) / saveparam[i - g_or_h][1]
            data.to_csv(datasavepath, index=0)
        return data


'''  datapath是数据来源，datasavepath是存储路径，这两个都是.csv的格式，method可以有两种选择方法，默认是输入字符串’mean‘，或者可以选择’median‘'''


def padingnan(datapath, datasavepath, method):
    data = pd.read_csv(datapath)
    data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
    if method == 'mean':
        data = data.fillna(data.mean())
    elif method == 'median':
        data = data.fillna(data.median())
    else:
        data = data.fillna(data.mean())
    data.to_csv(datasavepath, index=0)
    return data
