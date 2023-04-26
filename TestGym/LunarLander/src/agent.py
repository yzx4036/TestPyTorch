""" Create DQNAgent Class """
import os
import time
import numpy as np
import torch
import torch as T
from replay_buffer import ReplayBuffer
from network import DeepQNetwork
from utils import load_config

config = load_config("../config/config.yaml")


class DDQNAgent:
    def __init__(self, input_dims, n_actions, lr, discount_factor, eps, eps_dec, eps_min, batch_size,
                 replace, mem_size, algo=None, env_name=None, chkpt_dir=None, disappointing_score=-20,
                 disappointing_keep_going_ratio=0.5, disappointing_keep_going_max_count=20, disappointing_time=3):
        self.start_time = 0
        self.is_start = False
        self.is_keep_going = False  # 是否坚持挣扎
        self.is_keep_going_count = 0  # 坚持挣扎的次数
        self.disappointing_time = disappointing_time  # 失望的时间
        self.disappointing_keep_going_max_count = disappointing_keep_going_max_count
        self.input_dims = input_dims  # input dimensions 输入维度
        self.n_actions = n_actions  # number of actions 动作的个数
        self.action_space = [i for i in range(n_actions)]  # action space 动作空间
        self.lr = lr  # learning rate 学习率
        self.gamma = discount_factor  # discount factor 折扣因子，gamma的值越大，对未来奖励的重视程度就越高，因此智能体在做决策时更加关注未来的奖励。通常情况下，gamma的取值范围在0.9到0.99之间，可以根据具体问题进行调整。
        self.eps = eps  # epsilon-greedy 探索率
        self.eps_dec = eps_dec  # epsilon decay rate 探索率衰减率
        self.eps_min = eps_min  # minimum epsilon 最小探索率
        self.batch_size = batch_size  # batch size 批次大小
        self.replace_target_cnt = replace  # replace target network counter 替换目标网络计数器, 相当于多少步更新一次目标网络
        self.algo = algo  # algorithm 算法名
        self.env_name = env_name  # environment name 环境名
        self.chkpt_dir = chkpt_dir  # checkpoint directory 检查点目录
        self.learn_step_cntr = 0  # learning step counter 学习步计数器

        self.disappointing_score = disappointing_score  # disappointing score 不满意的分数
        self.disappointing_keep_ratio = disappointing_keep_going_ratio  # disappointing_keep_going_ratio 出现不满意分数时坚持的几率
        self.disappointing_keep_ratio_delta = 0  # disappointing_keep_going_ratio 出现不满意分数时坚持的几率的变化量

        self.last_reward = -100000  # 上一次的奖励

        print("失望分数: {} 失望时挣扎坚持的几率：{}".format(self.disappointing_score, self.disappointing_keep_ratio))

        self.memory = ReplayBuffer(mem_size, input_dims)

        # 创建Q网络和目标网络
        self.q_policy = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                     fc1_dims=config["fc1_dims"], fc2_dims=config["fc2_dims"],
                                     name=self.env_name + "_" + self.algo + "_q_policy")

        self.q_target = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                     fc1_dims=config["fc1_dims"], fc2_dims=config["fc2_dims"],
                                     name=self.env_name + "_" + self.algo + "_q_target")

    def store_transition(self, state, action, reward, new_state, done):
        self.last_reward = reward
        # 保存当前的state, action, reward, next_state, done到记忆库中
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action_from_nn(self, observation):
        # convert observation to pytorch tensor 转换成pytorch张量
        state = T.tensor([observation]).to(self.q_policy.device)
        # predict q-values for current state with policy network 预测当前状态的q值
        actions = self.q_policy.forward(state)
        # choose action with highest q-value 选择q值最大的动作
        action = T.argmax(actions).item()
        return action

    def choose_action(self, observation, current_score):
        if self.is_keep_going_count > self.disappointing_keep_going_max_count:
            return 0
        if self.is_keep_going:
            return 0
        _random = np.random.random()
        # 当前的探索率大于随机数时，随机选择一个动作，否则选择最优动作
        if np.random.random() < self.eps:
            # choose random action from action space 从动作空间中随机选择一个动作
            action = np.random.choice(self.action_space)
        elif current_score < self.disappointing_score:
            _random = np.random.random()
            if (_random > self.disappointing_keep_ratio):
                if self.is_keep_going_count > self.disappointing_keep_going_max_count * 0.5:
                    action = np.random.choice(self.action_space)
                else:
                    return 0
                # print("失望放弃！！_random：{} self.disappointing_keep_ratio：{} 分数：{}".format(_random,
                #                                                                              self.disappointing_keep_ratio,
                #                                                                              current_score))
            else:
                action = self.choose_action_from_nn(observation)
                self.is_keep_going = True
                self.is_keep_going_count += 1
                # print("失望坚持！！_random：{} self.disappointing_keep_ratio：{} 分数：{}".format(_random,
                #                                                                              self.disappointing_keep_ratio,
                #                                                                              current_score))
        else:
            action = self.choose_action_from_nn(observation)

        return action

    def replace_target_network(self):
        # check if learn step counter is equal to replace target network counter 检查学习步计数器是否等于替换目标网络计数器
        if self.learn_step_cntr % self.replace_target_cnt == 0:
            # load weights of policy network and feed them into target network 把策略网络的权重加载到目标网络中
            self.q_target.load_state_dict(self.q_policy.state_dict())
            return True
        return False

    def decrement_epsilon(self):
        # check if current epsilon is still greater than epsilon min 检查当前的探索率是否大于最小探索率
        if self.eps > self.eps_min:
            # decrement epsilon by epsilon decay rate 通过探索率衰减率来衰减探索率
            self.eps = self.eps - self.eps_dec
        else:
            # set epsilon to epsilon min 把探索率设置为最小探索率
            self.eps = self.eps_min

    # 采样经验缓存中的数据
    def sample_memory(self):
        # 从经验缓存中获取一个batch size的数据
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        # 把数据转换成tensor张量返回
        states = T.tensor(states).to(self.q_policy.device)
        actions = T.tensor(actions, dtype=T.long).to(self.q_policy.device)
        rewards = T.tensor(rewards).to(self.q_policy.device)
        new_states = T.tensor(new_states).to(self.q_policy.device)
        dones = T.tensor(dones).to(self.q_policy.device)

        return states, actions, rewards, new_states, dones

    def learn(self):
        # do not learn until memory size if greater or equal to batch size 保证记忆库中的数据大于batch size，小于batch size时不进行学习
        if self.memory.mem_cntr < self.batch_size:
            return None, None

        # set gradients to zero to do the parameter update correctly 设置梯度为0，以便正确地进行参数更新
        # PyTorch accumulates the gradients on subsequent backward passes PyTorch在后续的反向传播中累积梯度
        self.q_policy.optimizer.zero_grad()

        # replace target network 执行一次是否替换目标网络
        is_replace = self.replace_target_network()

        # create batch indices 创建批次索引
        batch_index = np.arange(self.batch_size)

        # get batch for training 进行一次采样获取一个batch的数据
        states, actions, rewards, new_states, dones = self.sample_memory()

        # compute q_values for each state, based on the selected action - Shape [batch_size, 1] 计算每个状态的q值，基于选择的动作和
        q_pred = self.q_policy.forward(states)[batch_index, actions]
        # compute q-values for each new_state with target network - Shape [batch_size, nb_actions] 计算目标网络中每个新状态的q值
        q_next = self.q_target.forward(new_states)

        # set q_next values for terminal states equals zero (no future reward if episode terminals) 设置终止状态的q_next值为0（如果episode终止，则没有未来奖励）
        q_next[dones] = 0.0

        # compute q-values of new_states with policy-network and store indices of best actions in max_actions - Shape [batch_size, nb_actions] 计算策略网络中每个新状态的q值，并将最佳动作的索引存储在max_actions中
        q_eval = self.q_policy.forward(new_states)  # Shape [batch_size, nb_actions] [32, 4]
        max_actions = T.argmax(q_eval, dim=1)

        # compute q-targets - Shape [batch_size, 1] 计算q-targets
        # best actions of q_next are choosen with the q_eval indices (max_actions) 最佳动作是通过q_eval索引（max_actions）选择的q_next
        q_target = rewards + self.gamma * q_next[batch_index, max_actions]
        # q_target = self.lr * (rewards + self.gamma * q_next[batch_index, max_actions])
        # q_target = self.lr * (rewards +  self.gamma * q_next[batch_index, max_actions]) + (1 - self.lr) * q_pred
        # dones_t = dones[batch_index]
        # print("rewards {}".format(rewards))
        # q_target = self.lr * (rewards + torch.where(dones, 2, self.gamma) * q_next[batch_index, max_actions]) +  torch.where(dones, (1 - self.lr) * q_pred, 0)

        # compute loss between q-targets and q-pred - Shape [batch_size, 1] 计算q-targets和q-pred之间的损失
        loss = self.q_policy.loss(q_target, q_pred).to(self.q_policy.device)

        # compute gradients 计算梯度 在深度学习中，我们通常使用反向传播算法计算损失函数对网络参数的导数，以便进行参数更新。这个过程称为反向传播。
        loss.backward()

        # perform optimization step (parameter update) 执行优化步骤（参数更新）
        self.q_policy.optimizer.step()

        # decrement epsilon by epsilon decay rate 通过探索率衰减率来衰减探索率
        self.decrement_epsilon()

        # increase learn step counter 增加学习步计数器
        self.learn_step_cntr += 1

        return loss, is_replace

    def mark_start(self, is_start):
        self.is_start = is_start
        self.start_time = time.time()
        self.is_keep_going_count = 0
        self.is_keep_going = False

    def check_time(self):
        time_now = time.time()
        if time_now - self.start_time > self.disappointing_time:
            self.is_keep_going = False
            return True, -50
        return False, 0

    def save_models(self):
        self.q_policy.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self, model_suffix=None):
        isSuccess = self.q_policy.load_checkpoint(model_suffix)
        self.q_target.load_checkpoint(model_suffix)
        # 
        # # 打印模型参数
        # for name, param in self.q_policy.named_parameters():
        #     print(name, param)
        # 
        return isSuccess
