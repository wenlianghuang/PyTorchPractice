import torch
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 定义数据集类别
class NumbersDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx, 0]
        y = self.dataframe.iloc[idx, 1]
        return torch.tensor([x], dtype=torch.float32), torch.tensor([y], dtype=torch.float32)
    
dataset = NumbersDataset("numbers_training.csv")
train_set, test_set = train_test_split(dataset, test_size=0.2)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 256)  # 增加隐藏层
        self.fc2 = nn.Linear(256, 128)  # 增加隐藏层
        self.dropout1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)  # 增加隐藏层
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)  # 增加隐藏层
        self.dropout3 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, 16)  # 增加隐藏层
        self.dropout4 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x

model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # 计算平均损失
    average_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

print('训练完成')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'准确率: {100 * correct / total}%')
