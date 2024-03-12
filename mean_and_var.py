import csv
import numpy as np
import os


file_path = './acc/ResGCN_ACC/Citeseer_no_track_32 .csv'

# 从 CSV 文件中读取测试准确率并计算平均值和方差

with open(file_path, mode='r') as csvfile:

    reader = csv.reader(csvfile)

    # 跳过表头（如果有的话）
    next(reader, None)

    # 存储所有数据的列表
    data = []

    # 遍历文件的每一行
    for row in reader:
        # 假设数据在第一列
        test_accuracy = float(row[0])  # 将数据从字符串转换为浮点数
        data.append(test_accuracy)
print(data)
mean_accuracy = np.mean(data)
variance_accuracy = np.std(data)

print(f"平均准确率: {mean_accuracy:.4f}")
print(f"准确率标准差: {variance_accuracy:.4f}")
data1 = [0.692, 0.605, 0.623, 0.608, 0.667, 0.7, 0.705, 0.695, 0.69, 0.704, 0.694,
         0.695, 0.709, 0.679, 0.701, 0.588, 0.424, 0.496, 0.702, 0.419, 0.691, 0.467, 0.687, 0.64, 0.182, 0.688, 0.722,
         0.544, 0.69, 0.698, 0.545, 0.633, 0.574, 0.181, 0.683, 0.709, 0.409, 0.722, 0.707, 0.578, 0.64, 0.692, 0.685,
         0.67]

mean_accuracy = np.mean(data1)
variance_accuracy = np.std(data1)

print(f"平均准确率: {mean_accuracy:.4f}")
print(f"准确率标准差: {variance_accuracy:.4f}")