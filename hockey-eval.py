
import numpy as np
import laserhockey.hockey_env as h_env
import gym
from importlib import reload
import time
import argparse
import datetime
import gym
import itertools
import torch
from sac.sac_better import SAC
from torch.utils.tensorboard import SummaryWriter
from sac.prio_replay_memory import PrioritizedReplay
from sac.replay_memory import ReplayMemory
import copy 
import os

parser = argparse.ArgumentParser(description='Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hockey")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--gamma', type=float, default=0.95, metavar='G')
parser.add_argument('--tau', type=float, default=0.005, metavar='G')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G')
parser.add_argument('--seed', type=int, default=111111, metavar='N')
parser.add_argument('--batch_size', type=int, default=4, metavar='N')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N')

args = parser.parse_args()

args.cuda =True if torch.cuda.is_available() else False

env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent

root = 'finals/'
runs = sorted(os.listdir(root))
runs = [r+'/' for r in runs]
agent = SAC(env.observation_space.shape[0], env.action_space, args)
opponent = SAC(env.observation_space.shape[0], env.action_space, args)

player1= 1
player2 =3
basic1 = False
basic2 = False
# print(f"{runs[player1]} vs {runs[player2]}")
print(f"{runs[player1]} vs {runs[player2]}")
models1 = sorted(os.listdir(root+runs[player1]))
actor = root+runs[player1]+models1[0]
critic = root+runs[player1]+models1[1]
target = root+runs[player1]+models1[2] if len(models1)==3 else None

models2 = sorted(os.listdir(root+runs[player2]))
o_actor = root+runs[player2]+models2[0]
o_critic = root+runs[player2]+models2[1]
o_target = root+runs[player2]+models2[2] if len(models2)==3 else None

agent.load_model(actor,critic,target)
opponent.load_model(o_actor,o_critic,o_target)


basic = h_env.BasicOpponent(weak=False)

time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#Tesnorboard
# writer = SummaryWriter(f"hockey-runs-defence/{time_}_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}")

# Memory
# memory = PrioritizedReplay(args.replay_size)
memory = ReplayMemory(args.replay_size,args.seed)

# Training Loop
total_numsteps = 0
updates = 0


o = env.reset()
# _ = env.render()

score_we = 0
score_they = 0

avg_reward = 0.
episodes = 1000
for i_episode  in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    won = None
    print(f"Round {i_episode}")
    while not done:
        obs_agent2 = env.obs_agent_two()

        if basic1:
            action = basic.act(state)
        else:
            action = agent.select_action(state, evaluate=True)
        
        if basic2:
            a2 = basic.act(obs_agent2)
        else:
            a2 = opponent.select_action(obs_agent2, evaluate=True)
        
        next_state, reward, done, info = env.step(np.hstack([action[0:4],a2[0:4]])) 
        env.render()
        episode_reward += reward
        won = info

        state = next_state
    avg_reward += episode_reward
    if info['winner']==1:
        score_we+=1
    elif info['winner']==-1:
        score_they+=1
    print(f"{score_we}:{score_they}")

avg_reward /= episodes


print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")

env.close()


