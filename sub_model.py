from matplotlib import pyplot as plt
from tensorflow.keras import models
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class IntermediateExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.sub = models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(n).output for n in layer_names]
        )

    def __call__(self, x):
        outs = self.sub(x, training=False)
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        return dict(zip(self.layer_names, outs))

import numpy as np
import matplotlib.pyplot as plt

def plot_feature_maps(feature_map, title='', max_channels=16):
    """
    兼容多种输入形状：
      (1, T, C) / (T, C) / (1, T) / (T,) / (1, 1, T, C) ...
    目标：得到形状为 (T, C) 的二维数组再画。
    """
    fm = np.asarray(feature_map)

    # 去掉所有 size 为 1 的前导维度（batch 维等）
    while fm.ndim > 2 and fm.shape[0] == 1:
        fm = fm[0]

    # 此时 fm.ndim 应为 2： (T, C)
    if fm.ndim == 3:
        # 例如 (N, T, C) 且 N>1，只取第一个样本
        fm = fm[0]
    elif fm.ndim == 1:
        # 例如 (T,)，视为单通道
        fm = fm[:, None]          # -> (T, 1)
    elif fm.ndim == 0:
        raise ValueError(f'特征图维度为 0，输入形状不对：{feature_map.shape}')

    assert fm.ndim == 2, f'期望 2 维特征图，实际 fm.shape={fm.shape}'

    T, C = fm.shape
    n_show = min(C, max_channels)

    fig, axes = plt.subplots(n_show, 1,
                             figsize=(10, 1.2 * n_show),
                             sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(fm[:, i], linewidth=1)
        ax.set_ylabel(f'ch{i}')
        ax.grid(alpha=0.3)
    axes[0].set_title(f'{title}  (T={T}, C={C})')
    plt.tight_layout()
    plt.show()


def to_2d(feature_map):
    """把任意常见形状规整为 (T, C) 的 2D 数组。"""
    fm = np.asarray(feature_map)

    # 去掉所有 batch 维（size==1 的前导维）
    while fm.ndim > 2 and fm.shape[0] == 1:
        fm = fm[0]

    if fm.ndim == 3:          # (N, T, C) N>1 → 取第一个样本
        fm = fm[0]
    elif fm.ndim == 1:        # (T,) → 单通道
        fm = fm[:, None]      # (T, 1)
    elif fm.ndim == 0:
        raise ValueError(f'无法处理 0 维输入：shape={feature_map.shape}')

    if fm.ndim != 2:
        raise ValueError(f'规整后仍是 {fm.ndim} 维：shape={fm.shape}')
    return fm                 # (T, C)


def plot_heatmap(feature_map, title='', max_channels=None):
    fm = to_2d(feature_map)   # (T, C)
    if max_channels is not None:
        fm = fm[:, :max_channels]

    fm = fm.T                 # (C, T)，imshow 行=步数、列=通道

    plt.figure(figsize=(10, max(2, 0.25 * fm.shape[0])))
    plt.imshow(fm, aspect='auto', origin='upper')
    plt.colorbar(label='activation')
    plt.xlabel('Time step')
    plt.ylabel('Channel')
    plt.title(f'{title}  (Channel={fm.shape[0]}, Step={fm.shape[1]})')
    plt.tight_layout()
    plt.show()

def plot_activation_overlay(feature_map, title='',
                            max_channels=None, linewidth=0.8, alpha=0.4):
    """
    每个通道单独 min-max 归一化到 [0,1] 后叠加画在同一张图上。
    """
    fm = to_2d(feature_map)               # (T, C)
    if max_channels is not None:
        fm = fm[:, :max_channels]

    T, C = fm.shape
    # 若 C==1，也可以退化为普通单通道曲线
    if C == 0:
        raise ValueError('特征图通道数为 0')

    # 逐通道 min-max 归一化（带 eps 防 0 除）
    vmin = fm.min(axis=0, keepdims=True)
    vmax = fm.max(axis=0, keepdims=True)
    fm_norm = (fm - vmin) / (vmax - vmin + 1e-8)

    plt.figure(figsize=(12, 4))
    for c in range(C):
        plt.plot(fm_norm[:, c], alpha=alpha, linewidth=linewidth)
    plt.xlabel('time step')
    plt.ylabel('normalized activation')
    plt.title(f'{title}  (T={T}, C={C})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_summary(history, y_test: np.ndarray, y_predict: np.ndarray):
    # 计算指标
    r2 = r2_score(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    print(f"R² (决定系数): {r2:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")

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