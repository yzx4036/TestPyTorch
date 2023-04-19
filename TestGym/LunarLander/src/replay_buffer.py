import numpy as np


# 经验缓存类
class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        # 在经验缓存中创建状态，动作，奖励，新状态，是否完成的内存
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # 存储输入到经验缓存
    def store_transition(self, state, action, reward, new_states, done):
        # 计算经验缓存中第一个可用的索引
        index = self.mem_cntr % self.mem_size
        # 基于索引存储输入的转换
        self.state_memory[index] = state
        self.new_state_memory[index] = new_states
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        # 自增经验缓存下标
        self.mem_cntr += 1

    # 从经验缓存中随机抽取样本
    def sample_buffer(self, batch_size):
        # 获取当前经验缓存下标可用的最大值
        max_mem = min(self.mem_cntr, self.mem_size)
        # 从零到最大值中随机抽取batch size个数
        # 不重复采样
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal
