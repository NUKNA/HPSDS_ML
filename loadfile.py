import numpy as np
import os

def loadfile(data_path, name_pre, x_sequence, runtime=3):
    """
    :param data_path: 数据文件路径
    :param name_pre: 文件名前缀（x ppm前面的部分）
    :param x_sequence: 浓度梯度列表
    :param runtime: 每个浓度运行次数
    :return: data, label

    Example
    ----------
    loadfile(r"D:\\Coding\\MachineLearning\\20260611\\8m",
    '20260611_[1]REF[2]DAS_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg100_',
    [0.5, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000,4000, 5000, 6000, 8000, 10000],
    3)
    """
    os.chdir(data_path)
    name_mid = 'ppm_run'
    name_suf = '.txt'
    data = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(1) + name_suf)
    label = np.empty([1,1], 'float64')
    label[0][0] = x_sequence[0]
    if data.shape != 1:
        data = np.delete(data, 1, 1)
        for i in range(2,runtime+1):
            label = np.append(label, [x_sequence[0]])
            temp = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(i) + name_suf)
            temp = np.delete(temp, 1, 1)
            data = np.hstack((data, temp))
        for k in range(1,len(x_sequence)):
            for i in range(1,runtime+1):
                label = np.append(label, [x_sequence[k]])
                temp = np.loadtxt(name_pre + str(x_sequence[k]) + name_mid + str(i) + name_suf)
                temp = np.delete(temp, 1, 1)
                data = np.hstack((data, temp))
    else:
        for i in range(2,runtime):
            temp = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(i) + name_suf)
            data = np.hstack((data, temp))
        for k in range(1,len(x_sequence)):
            for i in range(1,runtime):
                temp = np.loadtxt(name_pre + str(x_sequence[k]) + name_mid + str(i) + name_suf)
                data = np.hstack((data, temp))
    return data, label

if __name__=='__main__':
    loadfile(r"D:\Coding\MachineLearning\20260611\8m",
             '20260611_[1]REF[2]DAS_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg100_',
             [0.5, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000,
                  4000, 5000, 6000, 8000, 10000],
             3)