""" Train Agent """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf info, warning and error messages are not printed
import gymnasium as gym
import logging.config
import numpy as np
from agent import DDQNAgent
import utils
import signal

config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")


def save_model(agent, score_list, avg_score_list, episode_list, epsilon_list, i=""):
    agent.save_models()
    logger.info("Save models {}".format(config["env_name"]))

    # plot simple learning curve
    utils.plot_learning_curve(score_list, avg_score_list, config["env_name"], i)

    # store training data and config to csv file
    utils.store_training_data(episode_list, score_list, avg_score_list, epsilon_list, config["env_name"])
    utils.store_training_config(config, config["env_name"])


def load_model(agent):
    isSuccess = agent.load_models()
    logger.info("Load models {}".format(config["env_name"], agent.total_i))
    return isSuccess, agent.total_i

def train():
    if config["render"]:
        env = gym.make(config["env_name"], render_mode="human")
    else:
        env = gym.make(config["env_name"])
    # 输入的维数就是环境观察空间的维数，输出的维数就是动作空间的n值
    agent = DDQNAgent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=config["learning_rate"],
                      discount_factor=config["discount_factor"], eps=config["eps"], eps_dec=config["eps_dec"],
                      eps_min=config["eps_min"], batch_size=config["batch_size"],
                      replace=config["replace_target_network_cntr"], mem_size=config["mem_size"],
                      algo="ddqn", env_name=config["env_name"])

    # lists for storing data
    episode_list = []
    score_list = []
    avg_score_list = []
    epsilon_list = []
    best_score = -np.inf

    isSuccess, total_i = load_model(agent)

    logger.info("Start training")

    # 注册信号处理函数
    def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        save_model(agent, score_list, avg_score_list, episode_list, epsilon_list, total_i)
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if isSuccess:
        episode = total_i
    else:
        episode = 0
        total_i = 0

    for episode in range(config["training_episodes"]):
        done = False
        score = 0
        observation, info = env.reset()
        while not done:
            if config["render"]:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        episode_list.append(episode)
        score_list.append(score)
        epsilon_list.append(agent.eps)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score

        total_i = episode
        logger.info('episode: {}, score: {}, avg_score: {}, best_score: {}, epsilon: {}'.format(episode, "%.2f" % score,
                                                                                                "%.2f" % avg_score,
                                                                                                "%.2f" % best_score,
                                                                                                "%.2f" % agent.eps))
    logger.info("Finish training")

    save_model(agent, score_list, avg_score_list, episode_list, epsilon_list)
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
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger("train")
    train()
