import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

# 定义函数,取前k个特征对图像进行还原
def get_image_feature(s, k):
    # 对于s只保留前k个特征值
    s_temp = np.zeros(s.shape[0])
    s_temp[0:k] = s[0:k]
    s = s_temp * np.identity(s.shape[0]) # 生成主对角特征方阵
    # p,s,q重构矩阵
    temp = np.dot(p, s)
    temp = np.dot(temp, q)
    plt.imshow(temp, cmap=plt.cm.gray, interpolation='nearest')
    plt.savefig('./images/113312_{}.jpg'.format(k))
    plt.show()
    print(A-temp)

# 加载图片
image = Image.open('./images/113312.jpg').convert('L')
image.save('./images/113312_L.jpg')
A = np.array(image)
print(A.shape)
# 显示原图像
plt.imshow(A, cmap=plt.cm.gray, interpolation='nearest')
plt.show()

# 对图像矩阵进行奇异值分解
p, s, q = svd(A, full_matrices=False)
# 取前k个特征对图像进行还原,k分别取1%,10%,50%
get_image_feature(s, 10)
get_image_feature(s, 108)
get_image_feature(s, 540)