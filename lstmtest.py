import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from bisect import bisect


# -------------- 数据加载 --------------
class _NPZDataset(torch.utils.data.Dataset):
    def __init__(self, root, sequence_length=50):
        """
        读取 .npz 文件，并构建用于训练的 Dataset。

        Args:
            root (str): .npz 文件所在目录
            sequence_length (int): 输入的时间序列长度
        """
        self.sequence_length = sequence_length
        self._files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")]
        self._files.sort()
        self._buffer = []
        self._cum_size = [0]
        self.load_next_buffer()

    def load_next_buffer(self):
        """ 加载文件到 buffer """
        self._buffer = []
        self._cum_size = [0]

        for f in tqdm(self._files, desc="Loading NPZ files", unit="file"):
            with np.load(f) as data:
                state = data['state']  # 3-5列 (假设有3列)
                obs = data['obs']  # 6-26列 (假设有21列)
                #

                self._buffer.append({'state': state, 'obs': obs})
                self._cum_size.append(self._cum_size[-1] + len(obs) - self.sequence_length)

    def __len__(self):
        return self._cum_size[-1]

    def __getitem__(self, idx):
        file_index = bisect(self._cum_size, idx) - 1
        seq_index = idx - self._cum_size[file_index]
        data = self._buffer[file_index]

        # 获取 input 和 target 序列
        input_seq = data['obs'][seq_index: seq_index + self.sequence_length-1]
        target_seq = data['obs'][seq_index+1: seq_index + self.sequence_length]

        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


def get_dataloader(root, sequence_length=50, batch_size=32, shuffle=True, num_workers=4):
    dataset = _NPZDataset(root, sequence_length)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# -------------- LSTM 模型 --------------
class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, output_size=10):
        """
        input_size: 输入维度 (14)
        hidden_size: LSTM 隐藏层大小
        num_layers: LSTM 层数
        output_size: 输出维度 (10)
        """
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: [batch_size, sequence_length, input_size]
        输出: [batch_size, sequence_length, output_size]
        """
        batch_size = x.shape[0]

        # 初始化 LSTM 的隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM 计算
        out, _ = self.lstm(x, (h0, c0))

        # 通过全连接层
        out = self.fc(out)

        return out


# -------------- 训练函数 --------------
def test_model(root, sequence_length=50, batch_size=32, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    dataloader = get_dataloader(root, sequence_length, batch_size)

    # 初始化模型
    model = LSTMTimeSeries(input_size=14, hidden_size=64, num_layers=2, output_size=14).to(device)

    # 损失函数 & 优化器
    criterion = nn.MSELoss()

    # 训练循环
    model.eval()
    for epoch in range(1):
        total_loss = 0
        mselist=np.zeros((100))
        guard = 0

        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            guard+=1

            inputs, targets = inputs.to(device), targets.to(device)
            currentinput = inputs[:, 0:100]
            for inneri in range(100):

                currentinput = model(currentinput)
                labels = targets[:, inneri:inneri + 100]
                loss = criterion(currentinput, labels)
                mselist[inneri] += loss.item()



        mselist = mselist / guard

        return mselist


# -------------- 运行训练 --------------
if __name__ == "__main__":
    root = "./testdata/"  # .npz 文件夹路径
    mselist = test_model(root, sequence_length=201, batch_size=32, num_epochs=50, learning_rate=0.001)
    mselistlist = []
    for i in range(100):
        mselistlist.append(mselist[i])
    print(mselist)
    print(mselistlist)
