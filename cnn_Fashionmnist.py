import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import time

# 1. 检查设备：查看是否可以用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 超参数设置 
batch_size = 64            #一次性喂给模型64张图
learning_rate = 0.001      #学习率，控制模型每次参数更新幅度的超参数
num_epochs = 10            #模型训练次数

# 3. 数据准备 
# 数据预处理：转换为张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

# 下载训练集和测试集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    #DataLoader打包数据

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 4. 定义模型 
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        # 卷积层1：输入1通道（灰度图），输出32通道，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层2：输入32通道，输出64通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层：2x2窗口，步长2（图片尺寸减半）
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：经过两次池化后，图片变成 7x7，64通道 → 64*7*7=3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：128 → 10（10个类别）
        self.fc2 = nn.Linear(128, 10)
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 第一次卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv1(x)))
        # 第二次卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平：把二维特征图拉成一维向量
        x = x.view(x.size(0), -1)
        # 全连接层 + 激活 + Dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # 输出层
        x = self.fc2(x)
        return x

model = FashionCNN().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 5. 损失函数和优化器
criterion = nn.CrossEntropyLoss()    #分类专用
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   #新手友好

# 6. 训练循环
print("\n开始训练...")
train_start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()   #提取张量转化为python浮点数
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每 100 个 batch 打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # 每个 epoch 结束后打印训练准确率
    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}] 完成, 训练准确率: {train_accuracy:.2f}%')

train_end_time = time.time()
print(f"\n训练完成! 总耗时: {train_end_time - train_start_time:.2f} 秒")

#  7. 测试模型
print("\n在测试集上评估...")
model.eval()   #eval()测试时使用
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'测试集准确率: {test_accuracy:.2f}%')

#  8. 可视化一个预测结果
print("\n显示一个预测示例...")
model.eval()
# 取测试集第一个 batch
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 在 CPU 上显示第一张图片
plt.figure(figsize=(4, 4))
plt.imshow(images[0].cpu().squeeze(), cmap='gray')
plt.title(f'True: {labels[0].item()}, Predicted: {predicted[0].item()}')
plt.axis('off')
plt.show()

print("所有步骤完成!")