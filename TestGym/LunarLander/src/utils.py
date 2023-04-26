import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os


def plot_learning_curve(score_list, avg_score_list, loss_list, avg_loss_list, env, i, time):
    plt.figure(figsize=(12, 8))
    plt.plot(np.array(score_list), label="Score")
    plt.plot(np.array(avg_score_list), label="Average Score")
    plt.plot(np.array(avg_loss_list), label="avg_loss_list")
    plt.ylabel("Score")
    plt.xlabel("Episodes")
    plt.grid(axis="both")
    plt.legend(loc="lower right")
    plt.title('Learning Curve')
    plt.savefig('../plots/{}_ddqn_learning_curve_{}_{}.png'.format(env, i, time))
    plt.close()


def load_training_data(env):
    df = None
    file_path = "../data/training/{}_ddqn_training_data.csv".format(env)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    return df


def store_training_data(episodes, scores, avg_scores, epsilons, loss_list, avg_loss_list, env):
    training_data_dict = {"episode": episodes, "score": scores, "avg_score": avg_scores, "epsilon": epsilons,
                          "loss": loss_list,
                          "avg_loss": avg_loss_list}
    df = pd.DataFrame(training_data_dict)
    df.to_csv("../data/training/{}_ddqn_training_data.csv".format(env))


def store_training_config(config, env):
    df = pd.DataFrame(config, index=[0])
    df.to_csv("../data/training/{}_ddqn_training_config.csv".format(env))


def load_config(config_name):
    with open(os.path.join(config_name)) as file:
        config = yaml.safe_load(file)
    return config


def load_logging_config(filename):
    with open(filename, "rt") as f:
        logging_config = yaml.safe_load(f.read())
        f.close()
    return logging_config
