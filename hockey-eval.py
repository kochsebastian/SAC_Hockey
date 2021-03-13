
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

parser = argparse.ArgumentParser(description='Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hockey",
                    help='Gym environment')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian (default: Gaussian)')
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
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

args = parser.parse_args()

args.cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
actor = "strongplay_models_alpha/sac_actor_hockeyStrongPRE_reward-9.590869141533414_episode-38500_batch_size-4_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-10000000_t-2021-03-12_07-25-54"
critic = "strongplay_models_alpha/sac_critic_hockeyStrongPRE_reward-9.590869141533414_episode-38500_batch_size-4_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-10000000_t-2021-03-12_07-25-54"
agent.load_model(actor,critic)

args.hidden_size=256
opponent = SAC(env.observation_space.shape[0], env.action_space, args)
o_actor = "selfplay_models/sac_actor_hockeySelf_reward-9.742666643080318_episode-132700_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_14-54-00"
o_critic = "selfplay_models/sac_critic_hockeySelf_reward-9.742666643080318_episode-132700_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_14-54-00"
opponent.load_model(o_actor,o_critic)


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
        a2 = agent.select_action(obs_agent2, evaluate=True)
        
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


