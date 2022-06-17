import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(w, b, x):
    return x*w + b

def loss(y_pred,y) :
    return (y_pred - y)**2

w_cor = np.arange(0.0, 4.0, 0.1)
b_cor = np.arange(-2.0, 2.1, 0.1)

# 此处直接使用矩阵进行计算
w, b = np.meshgrid(w_cor, b_cor)
mse = np.zeros(w.shape)

for x,y in zip(x_data,y_data):
    y_pred_val = forward(w, b, x)
    mse += loss(y_pred_val, y)
mse /= len(x_data)

h = plt.contourf(w, b, mse)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel(r'w', fontsize=20, color='cyan')
plt.ylabel(r'b', fontsize=20, color='cyan')
ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()
