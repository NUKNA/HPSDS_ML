# 1D-CNN
利用HPSDS和一维卷积神经网络反演气体浓度。

## 2026/9/17 Update
目前只有 **main.py** 和 **loadfile.py** 文件可直接运行。其他文件代码可做参考，但不保证正确性。
 
## 2026/9/23 Update
- 修正了“当只有单列数据导入时，数据形状(Shape)不正确”的问题；
- 新增加函数"load_the_runtime"，用于导入单次runtime数据。

## 2026/9/29 Update.2
- 将复杂的模型设置移出，另设文件`model_config.py`

## 2026/9/29 Update
- 在`load_file.py`中新增函数`load_accord_txt`，简化了读取文件的流程；并且不用每次都修改代码，而是修改外部txt文件。

## 2026/10/8 Update
- 加入了将隐藏层可视化的功能（持续开发中……）。

## 2026/10/13 Update
- 将数据读取、归一化和裁切全部封装到`data_preprocess.py`中，删除文件`data_preprocess.py`。