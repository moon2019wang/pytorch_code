import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

print('Predict(before training)',4,forward(4))
for epoch in range(100):
    l = loss(1,2)
    for x,y in zip(x_data,y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data  = w.data -  0.01 * w.grad.data  # 0.01 learning rate
        w.grad.data.zero_()
    print('epoch:', epoch, l.item())
print('predict (after training)', 4, forward(4))
