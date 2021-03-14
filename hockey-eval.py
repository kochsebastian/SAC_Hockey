
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
from sac_better import SAC
from torch.utils.tensorboard import SummaryWriter
from prio_replay_memory import PrioritizedReplay
from replay_memory import ReplayMemory
import copy 

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

args.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# actor = "finals/tau1000/sac_actor_1000updates_hockey_reward-5.298743624008804_episode-80000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_07-21-03"
# critic = "finals/tau1000/sac_critic_1000updates_hockey_reward-5.298743624008804_episode-80000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_07-21-03"


opponent = SAC(env.observation_space.shape[0], env.action_space, args)

actor = "finals/alpha/sac_actor_500updates_hockey_reward--0.20085748776545811_episode-33000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.02_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_06-01-44"
critic = "finals/alpha/sac_critic_500updates_hockey_reward--0.20085748776545811_episode-33000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.02_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_06-01-44"
o_actor = "finals/pre/sac_actor_500updates_hockey_prio_reward-1.983261894330191_episode-20000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_08-28-09"
o_critic = "finals/pre/sac_critic_500updates_hockey_prio_reward-1.983261894330191_episode-20000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_08-28-09"
o_target = "finals/pre/sac_critic_target_500updates_hockey_prio_reward-4.514257658674091_episode-19500_batch_size-8_gamma-0.97_tau.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_07-48-40"
agent.load_model(actor,critic)
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
for _  in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False
    won = None
    while not done:

        # action = basic.act(state)
        action = agent.select_action(state, evaluate=True)
        obs_agent2 = env.obs_agent_two()
        # a2 = basic.act(obs_agent2)
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


