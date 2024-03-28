import torch
import torch.nn as nn 
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
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

# 加载模型的权重进行预测
def predict(input_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = Net()  # 创建模型实例
    loaded_model.load_state_dict(torch.load('model.pth', map_location=device))  # 加载权重
    loaded_model.eval()

    with torch.no_grad():
        output = loaded_model(input_data)
        _, predicted_label = torch.max(output, dim=1)
        predicted_label = predicted_label.item() + 1  # 将张量转换为标量，并映射回原始标签值
        return predicted_label

mytext = ''
with open("../request_res.txt",'r') as file:
    mytext = file.readline()

# 示例输入数据
sample_input = torch.tensor([int(mytext)], dtype=torch.float32).unsqueeze(0)  # 添加unsqueeze(0)将输入形状改为 [1, 1]
predicted_label = predict(sample_input)
#print(f'输入值为 1193 时的预测标签为: {predicted_label}')
with open('Direct_result.json','w') as file:
    if predicted_label == 1:
        file.write("Speed Number One")
    elif predicted_label == 2:
        file.write("Speed Number Two")
    elif predicted_label == 3:
        file.write("Speed Number Three")
    elif predicted_label == 4:
        file.write("Speed Number Four")
    elif predicted_label == 5:
        file.write("Speed Number Five")
    elif predicted_label == 6:
        file.write("Speed Number Six")
    elif predicted_label == 7:
        file.write("Speed Number Seven")
    elif predicted_label == 8:
        file.write("Speed Number Eight")
    elif predicted_label == 9:
        file.write("Speed Number Nine")
    elif predicted_label == 10:
        file.write("Speed Number Ten")
    elif predicted_label == 11:
        file.write("Speed Last")
