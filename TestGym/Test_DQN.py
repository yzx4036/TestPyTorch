import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym


## DQN的基本模板，使用两个神经网络共同作用，一个是eval_net，一个是target_net

# 超参数
BATCH_SIZE = 32             # 批处理大小
LR = 0.01                   # learning rate
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 2000      # 记忆库大小


env = gym.make("Blackjack-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print("Observation: ", observation, "Reward: ", reward, "Terminated: ", terminated, "Truncated: ", truncated, "Info: ", info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# class DQN(object):
#     def __init__(self):
#         # 建立 target net 和 eval net 还有 memory
#
#     def choose_action(self, x):
#         # 根据环境观测值选择动作的机制
#         # return action
#
#     def store_transition(self, s, a, r, s_):
#         # 存储记忆
#
#     def learn(self):
#         # target 网络更新
#         # 学习记忆库中的记忆