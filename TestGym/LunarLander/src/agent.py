""" Create DQNAgent Class """

import numpy as np
import torch as T
from replay_buffer import ReplayBuffer
from network import DeepQNetwork
from utils import load_config

config = load_config("../config/config.yaml")


class DDQNAgent:
    def __init__(self, input_dims, n_actions, lr, discount_factor, eps, eps_dec, eps_min, batch_size,
                 replace, mem_size, algo=None, env_name=None, chkpt_dir=None):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.lr = lr
        self.gamma = discount_factor
        self.eps = eps
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.learn_step_cntr = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_policy = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                     fc1_dims=config["fc1_dims"], fc2_dims=config["fc2_dims"],
                                     name=self.env_name+"_"+self.algo+"_q_policy")

        self.q_target = DeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims,
                                     fc1_dims=config["fc1_dims"], fc2_dims=config["fc2_dims"],
                                     name=self.env_name+"_"+self.algo+"_q_target")

    def store_transition(self, state, action, reward, new_state, done):
        # store transition in replay buffer
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.eps:
            # choose random action from action space
            action = np.random.choice(self.action_space)
        else:
            # convert observation to pytorch tensor
            state = T.tensor([observation]).to(self.q_policy.device)
            # predict q-values for current state with policy network
            actions = self.q_policy.forward(state)
            # choose action with highest q-value
            action = T.argmax(actions).item()

        return action

    def replace_target_network(self):
        # check if learn step counter is equal to replace target network counter
        if self.learn_step_cntr % self.replace_target_cnt == 0:
            # load weights of policy network and feed them into target network
            self.q_target.load_state_dict(self.q_policy.state_dict())

    def decrement_epsilon(self):
        # check if current epsilon is still greater than epsilon min
        if self.eps > self.eps_min:
            # decrement epsilon by epsilon decay
            self.eps = self.eps - self.eps_dec
        else:
            # set epsilon to epsilon min
            self.eps = self.eps_min

    def sample_memory(self):
        # get batch of transitions from replay memory
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        # convert to pytorch tensors and send to device (necessary for pytorch)
        states = T.tensor(states).to(self.q_policy.device)
        actions = T.tensor(actions, dtype=T.long).to(self.q_policy.device)
        rewards = T.tensor(rewards).to(self.q_policy.device)
        new_states = T.tensor(new_states).to(self.q_policy.device)
        dones = T.tensor(dones).to(self.q_policy.device)

        return states, actions, rewards, new_states, dones

    def learn(self):
        # do not learn until memory size if greater or equal to batch size
        if self.memory.mem_cntr < self.batch_size:
            return

        # set gradients to zero to do the parameter update correctly
        # PyTorch accumulates the gradients on subsequent backward passes
        self.q_policy.optimizer.zero_grad()

        # replace target network
        self.replace_target_network()

        # create batch indices
        batch_index = np.arange(self.batch_size)

        # get batch for training
        states, actions, rewards, new_states, dones = self.sample_memory()

        # compute q_values for each state, based on the selected action - Shape [batch_size, 1]
        q_pred = self.q_policy.forward(states)[batch_index, actions]
        # compute q-values for each new_state with target network - Shape [batch_size, nb_actions]
        q_next = self.q_target.forward(new_states)

        # set q_next values for terminal states equals zero (no future reward if episode terminals)
        q_next[dones] = 0.0

        # compute q-values of new_states with policy-network and store indices of best actions in max_actions
        q_eval = self.q_policy.forward(new_states)  # Shape [batch_size, nb_actions]
        max_actions = T.argmax(q_eval, dim=1)

        # compute q-targets - Shape [batch_size, 1]
        # best actions of q_next are choosen with the q_eval indices (max_actions)
        q_target = rewards + self.gamma * q_next[batch_index, max_actions]

        # compute loss between q-targets and q-pred
        loss = self.q_policy.loss(q_target, q_pred).to(self.q_policy.device)

        # compute gradients
        loss.backward()

        # perform optimization step (parameter update)
        self.q_policy.optimizer.step()

        # decrement epsilon
        self.decrement_epsilon()

        # increase learn step counter
        self.learn_step_cntr += 1

        return loss

    def save_models(self):
        self.q_policy.save_checkpoint()
        self.q_target.save_checkpoint()

    def load_models(self):
        self.q_policy.load_checkpoint()
        self.q_target.load_checkpoint()
