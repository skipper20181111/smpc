import numpy as np

'''
G端：任务发起方，拥有x值和y值
H端：联合训练方，拥有x值
'''


class lin_basic():
    '''
    初始化w，入参x    为x的矩阵(G,H两端都需要)(行数数据条数，列数是特征数+1)
    出参init_w为单列矩阵
    '''

    def init_w(x):
        init_w = np.random.rand(np.shape(x)[1], 1)
        return init_w

    '''
    wx值(单方预测值），（host端需要传递到guest端，根据两边wx之和算出残差）
    入参 x：x（特征）的矩阵
    入参w:w（上一次更新后的w值(单列矩阵)，如果第一次开始则为初始化的w值
    出参 wx:与x行数相等的一列
    '''

    def wx(x, w):
        wx = np.matmul(x, w)
        return wx

    '''
    残差，由G端计算产生，并传到H端
    入参：G_wx，G_wx，分别为两边的wx值,y为G端y值，参数数据格式全部为数据条数长的的单列矩阵
    出参residual：数据形式同入参
    '''

    def compute_d(wx_G, wx_H, y):
        residual = wx_G + wx_H - y
        return residual

    '''
    梯度，输入为残差d(一列矩阵)和x(多维矩阵)
    '''

    def gradient(residual, x):
        div_j = ((np.matmul(np.transpose(x), residual)) / (np.shape(x)[0] * np.shape(x)[1]))
        return div_j

    '''
    更新w值
    入参：旧的w，一列矩阵（长度x的列数）
    入参：div_j：一列矩阵（长度x的列数）
    学习率
    出参：更新后的w
    '''

    def update_w(w, div_j, alpha):
        w = w - alpha * div_j
        return w

    # 计算损失函数 d是残差
    def jtheta(d):
        return sum(d ** 2)


'''
收敛条件'''


class converged():
    # def __init__(self):
    _mean_of_what_ = 10
    m = 0
    m_0 = 0
    num = 0
    threshold = 0.0001

    def set_threshold(hold):
        converged.threshold = hold

    def init_orign():
        converged._mean_of_what_ = 10
        converged.m = 0
        converged.m_0 = 0
        converged.num = 0

    def mean_10(loss):
        converged.m_0 = converged.m
        converged.m = (1 - 1 / converged._mean_of_what_) * converged.m + (
                1 / converged._mean_of_what_) * loss

    def set_period(period):
        converged._mean_of_what_ = period

    def stop_or_ahead():
        if abs((converged.m_0 / converged.m) - 1) <= converged.threshold:
            converged.num += 1
        if converged.num >= 10:
            return True
        return False

    def jtheta(d):
        return sum(d ** 2)
