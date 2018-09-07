#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:54:31 2018

@author: zikunchen
"""

from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

SEED = 2018

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print('\n')
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.features = nn.Sequential(
            nn.Linear(input_size , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def init_weights(self):
        
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
        
        np.random.seed(SEED) 
        W0 = np.random.normal(0, 2/float(self.hidden_size + self.input_size), 
                              size=(self.hidden_size, self.input_size))
        
        np.random.seed(SEED) 
        b0 = np.random.normal(0, 1/float(self.hidden_size), 
                              size=self.hidden_size)
        
        np.random.seed(SEED) 
        W1 = np.random.normal(0, 2/float(self.hidden_size + self.output_size), 
                              size=(self.output_size, self.hidden_size))
        
        np.random.seed(SEED) 
        b1 = np.random.normal(0, 1/float(self.output_size), 
                              size=self.output_size)
        
        self.features[0].weight.data = torch.from_numpy(W0).type(dtype_float)
        self.features[0].bias.data = torch.from_numpy(b0).type(dtype_float)
        self.features[2].weight.data = torch.from_numpy(W1).type(dtype_float)
        self.features[2].bias.data = torch.from_numpy(b1).type(dtype_float)

    def forward(self, x):
        # TODO
        x = self.features(x)
        x = nn.functional.softmax(x, dim = -1)
        return x

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    # TODO
    reward_list = []
    for start in range(len(rewards)):
        reward_t = 0
        power = 0
        for t in range(start, len(rewards)):
            reward_t += gamma ** power * rewards[t]
            power += 1
        reward_list.append(reward_t)
    return reward_list

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0.1, 
            Environment.STATUS_INVALID_MOVE: -0.5,
            Environment.STATUS_WIN         : 10,
            Environment.STATUS_TIE         : 0,
            Environment.STATUS_LOSE        : -5
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    hidden_size = policy.features[0].weight.data.numpy().shape[0]
    best_rt = 0
    bsst_ep = 0
    episodes = list()
    avg_returns = list()
    
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0

    for i_episode in range(100000):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        count_invalid = 0
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == env.STATUS_INVALID_MOVE:
                count_invalid += 1
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            avg_rt = running_reward / log_interval
            print('Episode {}\tAverage return: {:.2f}\tInvalid Moves: {}'.format(
                i_episode, 
                avg_rt, 
                count_invalid))
            if avg_rt > best_rt:
                best_rt = avg_rt
                best_ep = i_episode
            
            episodes.append(i_episode)
            avg_returns.append(avg_rt)

            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "{}/policy-{}.pkl".format(hidden_size, i_episode))

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return episodes, avg_returns, best_ep, best_rt


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    hidden_size = policy.features[0].weight.data.numpy().shape[0]
    weights = torch.load("{}/policy-{}.pkl".format(hidden_size, episode))
    return policy.load_state_dict(weights)


if __name__ == '__main__':
    
########### Part 1 ###########   
    env = Environment()
    env.render()
    env.step(4)
    env.render()
    env.step(2)
    env.render()
    env.step(8)
    env.render()
    env.step(0)
    env.render()
    env.step(1)
    env.render()
    env.step(7)
    env.render()
    env.step(3)
    env.render()
    env.step(5)
    env.render()
    env.step(6)
    env.render()

########### Part 5 ###########     
    
# 5 a) training     

## DO NOT TRAIN IF WANT TO REPRODUCE THE OPTIMAL MODEL IN THE REPORT ##
    
#    policy = Policy(hidden_size = 55)
#    policy.init_weights()
#    env = Environment()
#    
#    episodes, avg_returns, best_ep, best_rt = train(policy, env, gamma=0.9)    
#        
#    plt.plot(episodes, avg_returns, color='blue')
#    plt.title('Training Curve')
#    plt.ylabel('Average Return')
#    plt.xlabel('Episode')
#    plt.show()
#    plt.figure()

# 5 d) Play 100 games against random and display 5 games
    
best_ep = 90000
random.seed(SEED)
torch.manual_seed(SEED)
env = Environment()
policy = Policy(hidden_size = 55)
load_weights(policy, best_ep)

win_count = 0
loss_count = 0
tie_count = 0
game_i = 1
game_list = [12, 27, 55, 58, 65]  #80

for i in range(100):
    state = env.reset()
    done = False
    
    if i in game_list:
        print('\nGame {} - No. {}'.format(game_i, i))
    
    while not done:
        action, logprob = select_action(policy, state)
        state, status, done = env.play_against_random(action)
        if i in game_list:
            env.render()
    if i in game_list:
        game_i += 1
        
    if status == 'win':
        win_count += 1
    if status == 'lose':
        loss_count += 1 
    if status == 'tie':
        tie_count += 1 
print('\nWin: {}\tLoss: {}\tTie: {}'.format(win_count, loss_count, tie_count))


########### Part 6 Win/Loss/Tie during training ###########  

random.seed(SEED)
torch.manual_seed(SEED)

policy = Policy(hidden_size = 55)

win_list = []
loss_list = []
tie_list = []
episodes = []

for i_episode in range(0, 100000, 1000):
    load_weights(policy, i_episode)
    win_count = 0
    loss_count = 0
    tie_count = 0
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
        if status == 'win':
            win_count += 1
        if status == 'lose':
            loss_count += 1 
        if status == 'tie':
            tie_count += 1 
            
    win_list.append(win_count)
    loss_list.append(loss_count)
    tie_list.append(tie_count)
    episodes.append(i_episode)
    print('Episode: {}\n'.format(i_episode))
    print('Win: {}\tLoss: {}\tTie: {}\n'.format(win_count, loss_count, tie_count))

# plot win/loss/tie curve
plt.plot(episodes, win_list, color='green')
plt.plot(episodes, loss_list, color='red')
plt.plot(episodes, tie_list, color='blue')
plt.title('Win/Loss/Tie Curve')
plt.ylabel('Win/Loss/Tie')
plt.xlabel('Episode')
plt.legend(['win', 'loss', 'tie'], loc='lower right')
plt.show()
plt.figure()



########### Part 7 - show first move distribution ###########  


# distribution of the first move 
load_weights(policy, 90000)

first_probs = Variable(first_move_distr(policy, env)).data.numpy().flatten()

for i in range(8):
    print (first_probs[i])


# show distribution throughout training
one_list = []
two_list = []
three_list = []
four_list = []
five_list = []
six_list = []
seven_list = []
eight_list = []
nine_list = []

for i_episode in range(0, 100000, 1000):
    load_weights(policy, i_episode)
    first_probs = Variable(first_move_distr(policy, env)).data.numpy().flatten()

    one_list.append(first_probs[0])
    two_list.append(first_probs[1])
    three_list.append(first_probs[2])
    four_list.append(first_probs[3])
    five_list.append(first_probs[4])
    six_list.append(first_probs[5])
    seven_list.append(first_probs[6])
    eight_list.append(first_probs[7])
    nine_list.append(first_probs[7])


#plot win/loss/tie curve
plt.plot(episodes, one_list, color='green')
plt.plot(episodes, two_list, color='red')
plt.plot(episodes, three_list, color='orange')
plt.plot(episodes, three_list, color='blue')
plt.plot(episodes, four_list, color='yellow')
plt.plot(episodes, five_list, color='black')
plt.plot(episodes, six_list, color='white')
plt.plot(episodes, seven_list, color='purple')
plt.plot(episodes, eight_list, color='magenta')

plt.title('First Move Probabilities')
plt.ylabel('Probability')
plt.xlabel('Episode')
plt.legend(['(1, 1)', '(1, 2)', '(1, 3)',
            '(2, 1)', '(2, 2)', '(2, 3)',
            '(3, 1)', '(3, 2)', '(3, 3)'
            ], loc='lower right')
plt.show()
plt.figure()



    