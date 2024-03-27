import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 檢查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義一個更複雜的神經網絡模型
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(10, 100)  # 增加至100個神經元
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 50)  # 增加至50個神經元
        self.fc3 = nn.Linear(50, 20)   # 增加至20個神經元
        self.fc4 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# 初始化模型並移至GPU
model = ComplexNN().to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 增加資料量
num_samples = 20000000  # 二千萬筆資料
input_data = torch.randn(num_samples, 10).to(device)
target_data = torch.randint(0, 5, (num_samples,)).to(device)

# 使用資料集和資料加載器
dataset = TensorDataset(input_data, target_data)
# 增加批次大小
data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)

# 訓練模型
num_epochs = 100  # 減少訓練迴圈次數，以免過早停止
for epoch in range(num_epochs):
    # 將模型設為訓練模式
    model.train()
    
    # 計算訓練資料集的損失
    total_loss = 0.0
    for inputs, targets in data_loader:
        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        
        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 計算並顯示每個迴圈的平均損失
    average_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}')

    

