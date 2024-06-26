import torch
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 检查是否有CUDA可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集类别
class NumbersDataset(Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx, 0]
        y = self.dataframe.iloc[idx, 1]
        return torch.tensor([x], dtype=torch.float32), torch.tensor(y - 1, dtype=torch.long)  # 修改为长整型，表示类别，并将标签值从1到6映射到0到5
    
dataset = NumbersDataset("numbers_training.csv")
train_set, test_set = train_test_split(dataset, test_size=0.2)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.2)
        self.fc6 = nn.Linear(16, 11)  # 修改为6类输出
    
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
        x = self.fc6(x)  # 不使用softmax，因为在交叉熵损失函数中包含了softmax
        return x

model = Net().to(device)  # 将模型移到GPU上
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU上
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

# 保存模型
torch.save(model.state_dict(), 'model.pth')
print('模型已保存')

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到GPU上
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)  # 修改为torch.max，并去掉.data
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'准确率: {100 * correct / total}%')

'''
# 加载模型进行预测
loaded_model = Net().to(device)  # 将加载的模型移到GPU上
loaded_model.load_state_dict(torch.load('model.pth'))
loaded_model.eval()

# 使用加载的模型进行预测
def predict(input_data):
    with torch.no_grad():
        output = loaded_model(input_data)
        _, predicted_label = torch.max(output, dim=1)
        predicted_label = predicted_label.item() + 1  # 将索引转换回从1到6的标签值
        return predicted_label

# 示例输入数据
sample_input = torch.tensor([[2021]], dtype=torch.float32, device=device)  # 添加device=device
predicted_label = predict(sample_input)
print(f'输入值为 2021 时的预测标签为: {predicted_label}')
'''
