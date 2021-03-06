import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset
batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# design model using class
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.L1 = torch.nn.Linear(784,512) # 样本为 28*28 的图片
        self.L2 = torch.nn.Linear(512,256)
        self.L3 = torch.nn.Linear(256, 128)
        self.L4 = torch.nn.Linear(128, 64)
        self.L5 = torch.nn.Linear(64, 10)

    def forward(self,x):
        x = x.view(-1,784) # -1其实就是自动获取mini_batch,转换成为一维向量
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = F.relu(self.L3(x))
        x = F.relu(self.L4(x))
        return self.L5(x) # 最后一层的激活函数为 softmax，使用 CrossEntropyLoss无需激活

model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum=0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f'  %(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

def predict():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1) # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # 张量之间的比较运算
    print('accuracy on test set: %d %%'  %(100*correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        predict()