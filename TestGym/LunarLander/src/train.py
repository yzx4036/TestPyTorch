""" Train Agent """

import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf info, warning and error messages are not printed
import gymnasium as gym
import logging.config
import numpy as np
from agent import DDQNAgent
import utils
import signal

config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")
np.random.seed(config["seed"])

def save_model(agent, score_list, avg_score_list, episode_list, epsilon_list, loss_list, avg_loss_list, i):
    agent.save_models()
    logger.info("Save models {}".format(config["env_name"]))

    current_time = time.strftime("%Y%m%d%H%M%S")
    # print("episode_list len={}".format(episode_list))
    # 使用plot采样曲线
    utils.plot_learning_curve(score_list, avg_score_list, loss_list, avg_loss_list, config["env_name"], i, current_time)

    # 保存训练数据到csv
    utils.store_training_data(episode_list, score_list, avg_score_list, epsilon_list, loss_list, avg_loss_list,
                              config["env_name"])
    utils.store_training_config(config, config["env_name"])


def load_model(agent, episode_list, score_list, avg_score_list, epsilon_list, loss_list, avg_loss_list):
    isSuccess = agent.load_models()
    training_data = utils.load_training_data(config["env_name"])
    if training_data is None:
        logger.info("Load training data {} failed".format(config["env_name"]))
    else:
        episode_list += training_data["episode"].tolist()
        score_list += training_data["score"].tolist()
        avg_score_list += training_data["avg_score"].tolist()
        epsilon_list += training_data["epsilon"].tolist()
        loss_list += training_data["loss"].tolist()
        avg_loss_list += training_data["avg_loss"].tolist()
        agent.eps = epsilon_list[len(episode_list) - 1] + np.random.random() * 0.1
        agent.eps = min(agent.eps, 1)
        agent.eps = max(agent.eps, agent.eps_min)

    logger.info("Load models {} success：{} i={}".format(config["env_name"], isSuccess, len(episode_list)))
    return isSuccess, len(episode_list)


def train():
    if config["render"]:
        env = gym.make(config["env_name"], render_mode="human")
    else:
        env = gym.make(config["env_name"])

    discount_factor = config["discount_factor"]
    # 输入的维数就是环境观察空间的维数，输出的维数就是动作空间的n值
    agent = DDQNAgent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=config["learning_rate"],
                      discount_factor=discount_factor, eps=config["eps"], eps_dec=config["eps_dec"],
                      eps_min=config["eps_min"], batch_size=config["batch_size"],
                      replace=config["replace_target_network_cntr"], mem_size=config["mem_size"],
                      algo="ddqn", env_name=config["env_name"], disappointing_score=config["disappointing_score"],
                      disappointing_keep_going_ratio=config["disappointing_keep_going_ratio"],
                      disappointing_keep_going_max_count=config["disappointing_keep_going_max_count"],
                      disappointing_time=config["disappointing_time"])

    # 定义一些列表，用于存储每轮的训练数据
    episode_list = []
    score_list = []
    avg_score_list = []
    epsilon_list = []
    loss_list = []
    avg_loss_list = []
    

    best_score = -np.inf
    best_loss = np.inf

    isSuccess, total_i = load_model(agent, episode_list, score_list, avg_score_list, epsilon_list, loss_list,
                                    avg_loss_list)
    if len(avg_score_list) > 0:
        best_score = max(avg_score_list)
    if len(avg_loss_list) > 0:
        a_list = [i for i in avg_loss_list if i != 0]
        best_loss = min(a_list)

    logger.info("Start training best_score={}, best_loss={}".format(best_score, best_loss))

    # 注册信号处理函数
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        save_model(agent, score_list, avg_score_list, episode_list, epsilon_list, loss_list, avg_loss_list, total_i)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if isSuccess:
        print("load model success, total_i={}".format(total_i))
        episode = total_i
    else:
        episode = 0
        total_i = 0

    # 开始训练， 从配置中获取训练轮数
    for episode in range(episode, episode + config["training_episodes"]):
        _episode_start_timestamp = time.time()
        # 每一轮初始化gym环境和分数
        done = False
        score = 0
        total_loss = 0
        observation, info = env.reset()
        frame_step = 0
        
        agent.mark_start(True)
        # 每一轮完整执行一次训练，累加当前轮中的分数，计算平均分数
        while not done:
            frame_step += 1
            if config["render"]:
                env.render()
            # 根据观察到的state选择动作
            action = agent.choose_action(observation, score)

            # todo 预处理动作

            # 执行动作，获取下一个新的observation_state，reward，done
            observation_, reward, done, truncated, info = env.step(action)

            # print("observation: {}, action: {}, reward: {}, observation_: {}, done: {}".format(observation, action, reward, observation_, done))

            # done, reward = utils.extra_reward(done, agent, score, reward , observation_, discount_factor)
        
            score += reward

            # 将每一步的当前state, action, reward, next_state, done存储到记忆库中
            # 个人理解：环境接收到动作后，会返回一个新的状态，这个新的状态就是下一个状态，所以这里的observation_就是下一个状态，相当于动作执行后所造成的影响和变化
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            loss, is_replace = agent.learn()

            if loss is not None:
                total_loss = np.sum(loss.item())
            if frame_step > config["max_frame_step"]:
                done = True

        episode_list.append(episode)
        score_list.append(score)
        avg_score = np.mean(score_list[-100:])  # 计算最近100轮的平均分数
        epsilon_list.append(agent.eps)
        avg_score_list.append(avg_score)
        loss_list.append(total_loss)
        avg_loss = np.mean(loss_list[-100:])  # 计算最近100轮的平均loss
        avg_loss_list.append(avg_loss)

        # 保存最好的分数
        if avg_score > best_score:
            best_score = avg_score

        # 保存最好的loss
        if 0 < avg_loss < best_loss:
            best_loss = avg_loss

        total_i = episode

        _episode_end_timestamp = time.time()
        _cost_time = _episode_end_timestamp - _episode_start_timestamp

        textString = "episode: {}, 耗时：{}， score: {}, avg_score: {}, best_score: {}, avg_loss: {}, best_loss: {}, epsilon: {} KeepCount={} ".format(
            episode,
            "%.2f" % _cost_time,
            "%.2f" % score,
            "%.2f" % avg_score,
            "%.2f" % best_score,
            "%.2f" % avg_loss,
            "%.2f" % best_loss,
            "%.3f" % agent.eps, agent.is_keep_going_count)
        logger.info(textString)

    logger.info("Finish training")

    save_model(agent, score_list, avg_score_list, episode_list, epsilon_list, loss_list, avg_loss_list, total_i)
    # # save policy and target network
    # agent.save_models()
    # logger.info("Save models {}".format(config["env_name"], i))
    #
    # # plot simple learning curve
    # utils.plot_learning_curve(score_list, avg_score_list, config["env_name"], i)
    #
    # # store training data and config to csv file
    # utils.store_training_data(episode_list, score_list, avg_score_list, epsilon_list, config["env_name"], i)
    # utils.store_training_config(config, config["env_name"])


if __name__ == "__main__":
    np.random.seed(1)
    logging.config.dictConfig(logging_config) 
    logger = logging.getLogger("train")
    train()
