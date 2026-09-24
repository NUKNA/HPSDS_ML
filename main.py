from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import loadfile
import numpy as np

def normalization(data:np.ndarray, minmax=False):
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

# 加载数据
x_1, y_1 = loadfile.loadfile(r'D:\科研文件\xianhushuju\20260921',
                             '20260921_REF_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg100_',
                             [60,70,100], 35)
x_2, y_2 = loadfile.loadfile(r'D:\科研文件\xianhushuju\20260922',
                             '20260922_REF_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg50_',
                             [3,4,5,6,7,8,9,10,20,30,40,50,60,70,100,500,600,700,800,900,1000,2000,3000,4000], 35)
x_3, y_3 = loadfile.loadfile(r'D:\科研文件\xianhushuju\20260922',
                             '20260922_REF_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg100_',
                             [300,400], 35)
x_train = np.hstack((x_1, x_2, x_3))
y_train = np.hstack((y_1, y_2, y_3))

x_test, y_test = loadfile.load_the_runtime(r'D:\科研文件\xianhushuju\20260922',
                             '20260922_REF_MPC8m_NH3_1512nm_CtrlT17.9667_EOM10V_RF999MHz5MHz_Scan100Hz0mA100mA_Sample200kHz1k_avg50_',
                             [3,4,5,6,7,8,9,10,20,30,40,50,60,70,100,500,600,700,800,900,1000,2000,3000,4000], 36)

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
model = Sequential([
    Conv1D(32, kernel_size=16, strides=2, activation='relu', padding='same', input_shape=(800, 1)),
    Conv1D(64, 8, strides=1, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])
model.summary()

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

# 计算指标
r2 = r2_score(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mean_squared_error(y_test, y_predict))
print(f"R² (决定系数): {r2:.4f}")
print(f"MAE (平均绝对误差): {mae:.4f}")
print(f"RMSE (均方根误差): {rmse:.4f}")

mpl.rcParams['font.family'] = 'Times New Roman'
# 训练曲线
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training lost (MSE)')
plt.plot(history.history['val_loss'], label='Verification lost (MSE)')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 真实值 vs 预测值散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, alpha=0.6, edgecolors='k')
# 对角线
min_val = min(y_test.min(), y_predict.min())
max_val = max(y_test.max(), y_predict.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('Concentration(Real)/ppm')
plt.ylabel('Concentration(Model predict)/ppm')
plt.title(f'Result of prediction (R² = {r2:.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()

# 真实值预测值阶梯图
plt.figure(figsize=(8, 6))
plt.scatter(np.arange(len(y_predict)), y_predict, alpha=0.6, edgecolors='k')
plt.stairs(y_test)
plt.yscale('log')
plt.xlabel('Data points')
plt.ylabel('Concentration(Model predict)/ppm')
plt.title(f'Result of prediction (R² = {r2:.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()

# 额外: 预测残差分布
plt.figure(figsize=(8, 6))
residuals = y_test - y_predict
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Prediction residual/ppm')
plt.ylabel('Frequency')
plt.title(f'Residual distribution (AVG = {np.mean(residuals):.4f}, STD = {np.std(residuals):.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
residuals = (y_test - y_predict) / y_test * 100
plt.scatter(y_test, residuals, edgecolor='k', alpha=0.7)
plt.xscale('log')
plt.xlabel('Real concentration/ppm')
plt.ylabel('Residual/%')
plt.title(f'Residual distribution (AVG = {np.mean(residuals):.4f}, STD = {np.std(residuals):.4f})')
plt.grid(True)
plt.tight_layout()
plt.show()

print('End of the program.')