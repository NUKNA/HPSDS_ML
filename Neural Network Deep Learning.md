# Neural Network | Deep Learning

## 层 Layers

典型的神经网络由层(Layers)组成，有输入层、隐藏层、输出层三部分。其中隐藏层是神经网络执行运算的部分，包括卷积(Convolution)、池化(Pooling)等。每一层由多个神经元组成，每一个神经元可以看成一个函数运算，可以是卷积、也可以是平均。

## 卷积 Convolution

$$
(f\ast g)(x)=\int_{-\infty}^{\infty}f(t)g(x-t)dt
$$

数据通常是离散的，离散形式为：一个滑动的卷积核在数据上进行多项式运算并滑动。

$$
a_0^{(1)}=\sum_{i=1}^kw_ia_i=w_1a_1+w_2a_2+\cdots
$$

写成矩阵形式：

$$
\mathbf{a}^{(1)}=
\begin{bmatrix}a_0^{(1)}\\a_1^{(1)}\\\vdots\\a_n^{(1)}\end{bmatrix}=
\begin{bmatrix}
w_{0,0}&w_{0,1}&\cdots&w_{0,n}\\
w_{1,0}&w_{1,1}&\cdots&w_{1,n}\\
\vdots&\vdots&\ddots&\vdots\\
w_{n,0}&w_{n,1}&\cdots&w_{n,n}\\
\end{bmatrix}
\begin{bmatrix}a_0^{(0)}\\a_1^{(0)}\\\vdots\\a_n^{(0)}\end{bmatrix}=
\mathbf{Wa}^{(0)}
$$

**其中 $w_i$ 称为权重因子，是神经网络的参数(Parameters). 调整参数来使神经网络的结果更加准确。**

## 偏置和激活函数 Bias and activation

为了使神经元的输出更有意义，通常需要采用一些计算将结果落在特定的区间。可以采用偏置(bias)，即直接加或者减去某个常数值，这么做可以忽略某些低于阈值的结果。还可以采用特定的激活函数(activation function)，对数据做处理。比如ReLU函数：

$$
ReLU(x)=\left\{
\begin{align*}
0&, x\le0\\
x&, x>0
\end{align*}
\right.
$$

## 损失函数 Cost function

损失函数(Cost function)用于量化预测结果和真实结果之间的差异。损失函数的值越小，说明模型预测越接近真实值，模型性能越好。在识别0~9的手写数字中，损失函数可以表示为：

![](https://thumbnail1.baidupcs.com/thumbnail/5ee669304vb4f40d4f14677f75edfa34?fid=272933994-250528-853649861054213&rt=pr&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-NC%2fg%2bQxrD5cm4wCPcbU1XeW0JEA%3d&expires=8h&chkbd=0&chkv=0&dp-logid=211841040140241656&dp-callid=0&time=1789459200&size=c2048_u1152&quality=90&vuk=272933994&ft=image&autopolicy=1)

连续型变量可以采用均方误差, $1-R^2$ 等作为损失函数。

损失函数的输入为模型的参数（可高达上万个），输出为一个，即损失值。损失函数值越小，模型越准确，这给我们调整参数提供了方向。

## 梯度下降法

通过计算损失函数的梯度，找到使损失函数值下降最快的方向，并以此指导参数调整的方法就是梯度下降法。梯度会越来越小，直至梯度为0，此时我们已找到模型参数的局部最优解。
