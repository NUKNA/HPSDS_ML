import numpy as np
import os

name_mid = 'ppm_run'
name_suf = '.txt'

def loadfile(data_path, name_pre, x_sequence, runtime=3):
    """
    读取一个浓度梯度下的数据文件。这些文件的路径、前缀（x ppm前面的部分）应该相同。提供浓度梯度和每个浓度运行次数。

    :param data_path: 数据文件路径
    :param name_pre: 文件名前缀（x ppm前面的部分）
    :param x_sequence: 浓度梯度列表
    :param runtime: 每个浓度运行次数（默认为3）
    :return: data, label 对应数据的x, y

    Example
    ----------
    loadfile(r"D:\\Coding\\MachineLearning\\20260611\\8m",
    '20260611_[1]REF[2]DAS_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg100_',
    [0.5, 1, 3, 5, 8, 10, 20, 30, 40, 50, 60, 80, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 3000,4000, 5000, 6000, 8000, 10000],
    3)
    """
    os.chdir(data_path)
    data = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(1) + name_suf)
    label = np.empty([1,1], 'float64')
    label[0][0] = x_sequence[0]
    if data.ndim != 1:
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
        data = data.reshape(-1,1)
        for i in range(2,runtime):
            label = np.append(label, [x_sequence[0]])
            temp = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(i) + name_suf)
            temp = temp.reshape(-1,1)
            data = np.hstack((data, temp))
        for k in range(1,len(x_sequence)):
            for i in range(1,runtime):
                label = np.append(label, [x_sequence[k]])
                temp = np.loadtxt(name_pre + str(x_sequence[k]) + name_mid + str(i) + name_suf)
                temp = temp.reshape(-1, 1)
                data = np.hstack((data, temp))
    return data, label

def load_the_runtime(data_path, name_pre, x_sequence, run_number):
    """
    读取一个浓度梯度下的单次运行文件。这些文件的路径、前缀（x ppm前面的部分）应该相同。提供浓度梯度和指定的运行序号。

    :param data_path: 数据文件路径
    :param name_pre: 文件名前缀（x ppm前面的部分）
    :param x_sequence: 浓度梯度列表
    :param run_number: 指定的运行序号
    :return: data, label 对应x, y
    """
    os.chdir(data_path)
    data = np.loadtxt(name_pre + str(x_sequence[0]) + name_mid + str(run_number) + name_suf)
    label = np.empty([1,1], 'float64')
    label[0][0] = x_sequence[0]
    if data.ndim != 1:
        data = np.delete(data, 1, 1)
        for k in range(1,len(x_sequence)):
            label = np.append(label, [x_sequence[k]])
            temp = np.loadtxt(name_pre + str(x_sequence[k]) + name_mid + str(run_number) + name_suf)
            temp = np.delete(temp, 1, 1)
            data = np.hstack((data, temp))
    else:
        data = data.reshape(-1,1)
        for k in range(1,len(x_sequence)):
            label = np.append(label, [x_sequence[k]])
            temp = np.loadtxt(name_pre + str(x_sequence[k]) + name_mid + str(run_number) + name_suf)
            temp = temp.reshape(-1, 1)
            data = np.hstack((data, temp))
    return data, label

def loadfile_accord_txt(data_path, txt_file_name):
    """
    根据提供的文件索引列表来读取文件。列表的格式为：

    |数据文件路径|数据文件前缀名（x ppm前面的部分）|浓度梯度（不要带方括号）|运行次数（不填为3，必须都填或者都不填）|

    :param data_path: 文件索引列表文件位置
    :param txt_file_name: 文件索引列表txt文件名（需要带后缀.txt）
    :return: data, label 所有列表中的文件的x, y全部合并在一起
    """
    # 读取文件列表文件
    os.chdir(data_path)
    file_list = np.loadtxt(txt_file_name, dtype=str, encoding='UTF-8')

    # 若只有一行则reshape为(1,-1)
    if file_list.ndim == 1:
        file_list = file_list.reshape(1,-1)
    # print(file_list.shape)

    # 根据每一行的记录读取相应的文件
    data = 0
    label = 0
    # 第一行的数据
    x_sequence = []
    for j in file_list[0][2].split(','):
        try:
            x_sequence.append(int(j))
        except ValueError:
            x_sequence.append(float(j))
        if file_list.shape[1] == 4:
            data, label = loadfile(file_list[0][0], file_list[0][1], x_sequence, int(file_list[0][3]))
        else:
            data, label = loadfile(file_list[0][0], file_list[0][1], x_sequence)
    #之后行的数据
    for i in range(1,file_list.shape[0]):
        x_sequence = []
        for j in file_list[i][2].split(','):
            try:
                x_sequence.append(int(j))
            except ValueError:
                x_sequence.append(float(j))
        if file_list.shape[1] == 4:
            x_temp, y_temp = loadfile(file_list[i][0], file_list[i][1], x_sequence, int(file_list[i][3]))
            data = np.hstack((data, x_temp))
            label = np.hstack((label, y_temp))
        else:
            x_temp, y_temp = loadfile(file_list[i][0], file_list[i][1], x_sequence)
            data = np.hstack((data, x_temp))
            label = np.hstack((label, y_temp))

    return data, label

if __name__=='__main__':
    loadfile_accord_txt(r'D:\科研文件\xianhushuju','file_list.txt')