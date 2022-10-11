import pandas as pd

stcsv=pd.read_csv('E:/1.csv')

stcsv.sort_values(by='Active_Power',inplace=True,ascending=False)

print(stcsv)
stcsv.to_csv("E:/stcsv.csv", index=False)
import csv
import numpy as np

with open('E:/stcsv.csv') as csv_file:

    row = csv.reader(csv_file, delimiter=',')


    next(row)  # 读取首行
    Active_Power = []  # 建立一个数组来存储

    # 读取除首行之后每一行的数据，并将其加入到数组之中
    for r in row:
        Active_Power.append(float(r[3]))  # 将字符串数据转化为浮点型加入到数组之中
        print(np.var(Active_Power))  # 输出方差

print(np.mean(Active_Power))  # 输出均值
