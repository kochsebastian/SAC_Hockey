
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
# from prio_replay_memory import PrioritizedReplay
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hockey",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--train', action="store_true",
                    help='placeholder')
args = parser.parse_args()


env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model('full_player_models/sac_actor_hockeySelf_reward-9.753174685381822_episode-10000_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-08_15-32-17',"full_player_models/sac_critic_hockeySelf_reward-9.753174685381822_episode-10000_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-08_15-32-17")

# agent = SAC(env.observation_space.shape[0], env.action_space, args)
# agent.load_model('full_player_models/sac_actor_hockeyStrong_reward-9.43225949089355_episode-97200_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_02-13-47','full_player_models/sac_critic_hockeyStrong_reward-9.43225949089355_episode-97200_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_02-13-47')
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

        obs_agent2 = env.obs_agent_two()
        action =basic.act(state)
        a2 = agent.select_action(obs_agent2, evaluate=True)
        
        
        # a2 = opponent.select_action(obs_agent2,evaluate=True)
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


