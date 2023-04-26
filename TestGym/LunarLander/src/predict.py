""" Test Agent """

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tf info, warning and error messages are not printed
import gymnasium as gym
import logging
from logging import config
import numpy as np
from agent import DDQNAgent
import utils

config = utils.load_config("../config/config.yaml")
logging_config = utils.load_logging_config("../config/logging.yaml")


def predict():
    if config["render"]:
        env = gym.make(config["env_name"], render_mode="human")
    else:
        env = gym.make(config["env_name"])
    discount_factor = config["discount_factor"]
    agent = DDQNAgent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=config["learning_rate"],
                      discount_factor=discount_factor, eps=config["eps"],
                      eps_dec=config["eps_dec"], eps_min=config["eps_min"],
                      batch_size=config["batch_size"], replace=config["replace_target_network_cntr"],
                      mem_size=config["mem_size"], algo="ddqn", env_name=config["env_name"])

    # load pretrained models
    agent.load_models(config["test_predict_model_suffix_name"])
    # set epsilon to zero as we want to choose the predicted best action (highest q-value) and not a random action
    agent.eps = 0.0

    # lists for storing data
    score_list = []
    avg_score_list = []

    logger.info("Start testing")
    for i in range(config["test_episodes"]):
        done = False
        score = 0
        observation, info = env.reset()
        while not done:
            if config["render"]:
                env.render()
            action = agent.choose_action(observation, score)
            observation_, reward, done, truncated, info = env.step(action)
            if (done and score > -50):
                if reward < 0:
                    # print("done and reward src_reward{} new_reward{} 勉强".format(reward, reward + 10))
                    reward = reward + 10
                else:
                    # print("done and reward src_reward{} new_reward{}".format(reward, reward + 100 * (
                    #         discount_factor * np.max(observation_))))
                    reward = reward + 100 * (discount_factor * np.max(observation_))
            score += reward
            observation = observation_

        score_list.append(score)
        avg_score = np.mean(score_list[-100:])
        avg_score_list.append(avg_score)

        logger.info('episode: {}, score: {}, avg_score: {}, epsilon: {}'.format(i, "%.2f" % score, "%.2f" % avg_score,
                                                                                "%.3f" % agent.eps))

    logger.info("Finish testing")


if __name__ == "__main__":
    logging.config.dictConfig(logging_config)
    # print("logging_config: {}".format(logging_config))
    logger = logging.getLogger("test")
    predict()
