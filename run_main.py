import subprocess
import csv
import numpy as np
import config
import re

# 设置运行主程序的次数
num_runs = 50


# 运行主程序多次
for run in range(1, num_runs + 1):

    subprocess.run(["python", "main.py"])

file_path = config.Config.accuracy_save_path
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

mean_accuracy = np.mean(data)
standard_deviation = np.std(data)

print(f"平均准确率: {mean_accuracy:.4f}")
print(f"准确率标准差: {standard_deviation:.4f}")
