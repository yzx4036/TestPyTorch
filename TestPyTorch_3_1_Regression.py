import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.size())
y = x.pow(2) + 0.2 * torch.rand(x.size())


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 定义每层用什么样的形式
        # 建立一个隐藏层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        # 建立一个输出层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # 激活函数包装隐藏层作为predict的输入
        x = F.relu(self.hidden(x))
        x = self.predict(x)  # linear output
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)  # 隐藏层10个神经元
print(net)  # net architecture

# optimizer 是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

for t in range(10000):
    prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值
    loss = loss_func(prediction, y)  # 计算两者的误差
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

    # 接着上面来
    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.01)