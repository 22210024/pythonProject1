import cv2
import numpy as np
import matplotlib.pyplot as plt
def img2gray(im):
    im[:,:,0] = im[:,:,0]*0.3
    im[:,:,1] = im[:,:,1]*0.59
    im[:,:,2] = im[:,:,2]*0.11
    im = np.sum(im, axis=2)
    im = np.array(im,dtype=np.uint8)
    return im
def gaussian_filter(img,k_size,sigma):
    if len( img.shape ) == 3:
        print("错误，请输入灰度图")
    else:
        print("输入成功")
        H,W = img.shape
    kplus = k_size//2
    K = np.zeros((k_size, k_size), dtype=np.float64)
    y, x = np.ogrid[-kplus:-kplus+k_size, -kplus:-kplus+k_size]
    K[y + 1, x + 1] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    new = np.zeros((kplus*2+H,kplus*2+W),dtype=np.float64)
    new[kplus:kplus+H,kplus:kplus+W] = img.copy().astype(np.float64)
    tmp = new.copy()

    for h in range(H):
        for w in range(W):
                new[kplus + h, kplus + w] = np.sum(K*tmp[h:h+k_size,w:w+k_size])
    new = np.clip(new,0,255)
    new = new[kplus:kplus+H,kplus:kplus+W].astype(np.uint8)
    # cv.imshow("result", new)
    # cv.waitKey()
    return new

def gradient(img):
#sobel算子
    S_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    S_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    H,W = img.shape
#扩充图像边界
    img2=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE)
    tmp = img2.copy().astype(np.float64)
    I1 = np.zeros((H+2,W+2), dtype=np.float64)
    I2 = np.zeros((H+2,W+2), dtype=np.float64)
#卷积运算
    for h in range(H):
        for w in range(W):
                I1[h,w] = np.sum(S_x*tmp[h:h+3,w:w+3])
                I2[h,w] = np.sum(S_y*tmp[h:h+3, w:w+3])
    I1 = np.clip(I1,0,255)
    I2 = np.clip(I2,0,255)
# cv.imshow("i1",I1)
# cv.imshow("i2",I2)
    G = np.sqrt(I1*I1+I2*I2)
    theta = np.arctan2(I2,(I1+0.0000000000001))*180/np.pi
    return G,theta

def nm_suppression(G):
    I_copy = np.zeros(G.shape)
    G_plus = cv2.copyMakeBorder(G, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    anchor = np.where(G_plus != 0)

    for i in range(len(anchor[0])):
        x = anchor[0][i]
        y = anchor[1][i]
        alter_point = G[x, y]
        g1 = G[x - 1, y - 1]
        g2 = G[x + 1, y - 1]
        g3 = G[x - 1, y + 1]
        g4 = G[x + 1, y + 1]
        if g1 < alter_point and g2 < alter_point and g3 < alter_point and g4 < alter_point:
            I_copy[x, y] = alter_point

    # img_uint8 = I_copy.astype(np.uint8)
    return I_copy

def b_threshold(I_copy):
    I_dt = np.zeros(I_copy.shape)
    TL = 0.4*np.max(I_copy)
    TH = 0.5*np.max(I_copy)
    for i in range(1,I_copy.shape[0]-1):
        for j in range(1,I_copy.shape[1]-1):
            if I_copy[i,j]<TL:
                I_dt[i,j] = 0
            elif I_copy[i,j]>TH:
                I_dt[i, j] = 255
            elif  (( I_copy[i+1,j] > TH) or (I_copy[i-1,j] > TH )or( I_copy[i,j+1] > TH )or
                    (I_copy[i,j-1] > TH) or (I_copy[i-1, j-1] > TH )or ( I_copy[i-1, j+1] >TH) or
                    ( I_copy[i+1, j+1] > TH ) or ( I_copy[i+1, j-1] > TH) ):
                I_dt[i, j] = 255
    return I_dt
img = cv2.imread('E:/png/1.png')
im = np.array(img)
im_gray = img2gray(im)
new = gaussian_filter(im_gray,k_size=3,sigma=1.3)
# cv2.imshow("mohu",new)
G,theta = gradient(new)
I_copy = nm_suppression(G)
img_uint8 = I_copy.astype(np.uint8)
I_dt = b_threshold(I_copy)
#cv2.imshow("feijidazhi",img_uint8)
#cv2.imshow("zuizhongjieguo",I_dt)
cv2.waitKey()

plt.subplot(221)
plt.imshow(img), plt.title('yuantu')
plt.subplot(222)
plt.imshow(im_gray, 'gray'), plt.title('huidutu')
plt.subplot(223)
plt.imshow(I_copy, 'gray'), plt.title('feijidazhiyizhi')
plt.subplot(224)
plt.imshow(I_dt, 'gray'), plt.title('zuizhongxiaoguo')
plt.show()



