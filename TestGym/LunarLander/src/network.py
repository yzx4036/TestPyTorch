""" Create DeepQNetwork Class """

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims, fc2_dims1, fc2_dims2):
        super(DeepQNetwork, self).__init__()
        self.name = name
        self.checkpoint_file = os.path.join("../models/", name)
        self.fc1_dims = fc1_dims # number of neurons in first hidden layer 第一个隐藏层的神经元个数
        self.fc2_dims1 = fc2_dims1 # number of neurons in second hidden layer 第二个隐藏层的神经元个数
        self.fc2_dims2 = fc2_dims2 # number of neurons in second hidden layer 第二个隐藏层的神经元个数
        self.fc1 = nn.Linear(*input_dims, self.fc1_dims) # 创建第一个全连接层，输入维度为input_dims，输出维度为fc1_dims
        self.fc2_middle1 = nn.Linear(self.fc1_dims, self.fc2_dims1) # 创建第二个全连接层，输入维度为fc1_dims，输出维度为fc2_dims
        self.fc2_middle2 = nn.Linear(self.fc2_dims1, self.fc2_dims2) # 创建第二个全连接层，输入维度为fc1_dims，输出维度为fc2_dims
        self.fc3 = nn.Linear(self.fc2_dims2, n_actions) # 创建第三个全连接层，输入维度为fc2_dims，输出维度为n_actions
        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr) # 创建优化器，使用RMSprop优化器
        self.optimizer = optim.Adam(self.parameters(), lr=lr) # 创建优化器，使用SGD优化器
        self.loss = nn.MSELoss() # 创建损失函数，使用均方误差损失函数
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu') # 创建设备，使用GPU或CPU
        self.to(self.device) # 将网络模型加载到设备上

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2_middle1(x))
        x = F.relu(self.fc2_middle2(x))
        actions = self.fc3(x)

        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, model_suffix=None):
        if model_suffix is not None:
            self.checkpoint_file = self.checkpoint_file + "_" + model_suffix
        if os.path.exists(self.checkpoint_file):
            print("Loading checkpoint... 成功")
            self.load_state_dict(T.load(self.checkpoint_file))
            return True
        else:
            print("No checkpoint found... {}".format(self.checkpoint_file))
            return False