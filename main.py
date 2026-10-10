from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sub_model as sm
import loadfile
import model_config

def normalization(data:np.ndarray, minmax:bool=False):
    """
    输入一个数据序列，自动归一化。

    :param data: 输入数据序列，任意维度
    :return: 处理后的序列 ``data``, 序列的最小值 ``data_min`` 和最大值 ``data_max``
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
    for i in range(0, len(data)):
        data[i] = (data[i] - data_min) / (data_max - data_min)
    if minmax:
        return data, data_min, data_max
    else:
        return data

def denormalization(data:np.ndarray, data_min, data_max):
    """
    归一化的反向计算，逆归一化。

    :param data: 输入数据序列
    :param data_min: 归一化前数列的最小值
    :param data_max: 归一化前数列的最大值
    :return: 逆归一化后的数列
    """
    for i in range(0, len(data)):
        data[i] = (data_max - data_min) * data[i] + data_min
    return data

x_train, y_train = loadfile.loadfile_accord_txt(r'D:\科研文件\xianhushuju', 'train.txt')
x_test, y_test = loadfile.loadfile_accord_txt(r'D:\科研文件\xianhushuju', 'test.txt')


# 数据预处理
# 转置数据
x_train = x_train.transpose()
x_test = x_test.transpose()
# 删除吸收列
x_train = np.delete(x_train, range(0,200), axis=1)
x_test = np.delete(x_test, range(0,200), axis=1)
# 归一化处理
x_train = normalization(x_train)
x_test = normalization(x_test)
y_train, y_train_min, y_train_max = normalization(y_train, True)
y_test, y_test_min, y_test_max = normalization(y_test, True)

# 构建模型
model = model_config.build_model()

# 编译模型
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mse', 'mae', 'accuracy'])

# 训练模型
history = model.fit(x_train, y_train,
          batch_size=12,
          epochs=100,
          verbose=1,
          validation_data=(x_test, y_test))

# 评估模型
y_predict = model.predict(x_test)
y_predict = y_predict.flatten()

y_test = denormalization(y_test, y_test_min, y_test_max)
y_predict = denormalization(y_predict, y_test_min, y_test_max)

# 获取中间层的输出
mpl.rcParams['font.family'] = 'Times New Roman'
layer_names = ['conv1', 'conv2', 'maxpool', 'dropout1', 'flatten', 'dropout2', 'dense1']
extractor = models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(n).output for n in layer_names]
)
feats = extractor.predict(x_test, verbose=0)
for name, f in zip(layer_names, feats):
    # sm.plot_feature_maps(f, title=f'Feature maps of {name} (shape={f.shape})')
    sm.plot_heatmap(f, title=f'{name} activations')
    # sm.plot_activation_overlay(f, title=f'{name}(per-channel normalized)')

# 计算指标
r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print(f"R² (决定系数): {r2:.4f}")
print(f"MAE (平均绝对误差): {mae:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")

sm.plot_summary(history, y_test, y_predict, r2, mae, rmse)

print('End of the program.')