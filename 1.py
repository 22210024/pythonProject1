import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False
# 读取数据
c = pd.read_excel(r'E:/1.xlsx')
# 绘制单条折线图
plt.scatter(c.Global_Horizontal_Radiation,  # x轴数据
            c.Active_Power,  # y轴数据
            s=100,
            marker='.',
            )

# 对于X轴，只显示x中各个数对应的刻度值
plt.xticks(fontsize=8, )  # 改变x轴文字值的文字大小
# 添加y轴标签
plt.ylabel('功率')
plt.xlabel('辐照度')
# 添加图形标题
plt.title('数据')

plt.savefig(r'E:/png/1.png',dpi=900,bbox_inches='tight')
# 显示图形
plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('E:/png/1.png')
GrayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh1=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(GrayImage,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3=cv2.threshold(GrayImage,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(GrayImage,127,255,cv2.THRESH_TOZERO_INV)
titles = ['Gray Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [GrayImage, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
   plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()