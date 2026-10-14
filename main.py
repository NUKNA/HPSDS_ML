from tensorflow.keras import models
import matplotlib as mpl
import numpy as np
import sub_model as sm
import model_config as mdconfig
import data_preprocess as dp

# 读取数据
x_train, y_train = dp.loadfile_accord_txt(r'D:\科研文件\xianhushuju', 'train.txt')
x_test, y_test = dp.loadfile_accord_txt(r'D:\科研文件\xianhushuju', 'test.txt')

# 裁切数据
start = 300
end = 1000
x_train = dp.slice(x_train, start, end)
x_test = dp.slice(x_test, start, end)

# 归一化处理
x_train = dp.normalization(x_train)
x_test = dp.normalization(x_test)
y_train, y_train_min, y_train_max = dp.normalization(y_train, True)
y_test, y_test_min, y_test_max = dp.normalization(y_test, True)

# 构建模型
model = mdconfig.build_model(end-start)

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

y_test = dp.denormalization(y_test, y_test_min, y_test_max)
y_predict = dp.denormalization(y_predict, y_test_min, y_test_max)

# 获取中间层的输出
mpl.rcParams['font.family'] = 'Times New Roman'
layer_names = mdconfig.layer_name()
extractor = models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(n).output for n in layer_names]
)
feats = extractor.predict(x_test, verbose=0)
for name, f in zip(layer_names, feats):
    # sm.plot_feature_maps(f, title=f'Feature maps of {name} (shape={f.shape})')
    sm.plot_heatmap(f, title=f'{name} activations')
    # sm.plot_activation_overlay(f, title=f'{name}(per-channel normalized)')

sm.plot_summary(history, y_test, y_predict)

print('End of the program.')