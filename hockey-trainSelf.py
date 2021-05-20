
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
import random

import os

parser = argparse.ArgumentParser(description='Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Hockey")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--gamma', type=float, default=0.97, metavar='G')
parser.add_argument('--tau', type=float, default=0.005, metavar='G')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G')
parser.add_argument('--alpha', type=float, default=0.01, metavar='G')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G')
parser.add_argument('--seed', type=int, default=111111, metavar='N')
parser.add_argument('--batch_size', type=int, default=8, metavar='N')
parser.add_argument('--num_steps', type=int, default=10000001, metavar='N')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N')

args = parser.parse_args()

args.cuda =True if torch.cuda.is_available() else False


env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
actor = "competionW/sac_actor_all_win_reward-7.048826588216617_episode-2500_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-16_02-30-46"
critic = "competionW/sac_critic_all_win_reward-7.048826588216617_episode-2500_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-16_02-30-46"
target = "competionW/sac_target_all_win_reward-7.048826588216617_episode-2500_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-16_02-30-46"
agent.load_model(actor,critic)
opponent = copy.deepcopy(agent)

root = 'finals/'
runs = sorted(os.listdir(root))
runs = [r+'/' for r in runs]

opponents = [SAC(env.observation_space.shape[0], env.action_space, args) for i in range(len(runs))]


for i in range(len(opponents)):
    models2 = sorted(os.listdir(root+runs[i]))
    o_actor = root+runs[i]+models2[0]
    o_critic = root+runs[i]+models2[1]
    o_target = root+runs[i]+models2[2] if len(models2)==3 else None
    opponents[i].load_model(o_actor,o_critic,o_target)

basic_strong = h_env.BasicOpponent(weak=False)
time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#Tesnorboard
writer = SummaryWriter(f"selfplay-comp/all_win{time_}_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}")

# Memory
memory = PrioritizedReplay(args.replay_size)
# memory = ReplayMemory(args.replay_size,args.seed)

# Training Loop
total_numsteps = 0
updates = 0
update_now = False


o = env.reset()
# _ = env.render()
last_avg=0
save_now = False 
threshold_avg = 0
number_avg=0
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    opp = random.randint(-1, len(opponents)-1)
    while not done:
        # state = env.obs_agent_two()
        if args.start_steps > total_numsteps or random.random() < 0.1:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, memory_ = agent.update_parameters(memory, args.batch_size, updates)
                memory=memory_

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        # a2 = [10,0.,0,0] 
        obs_agent2 = env.obs_agent_two()
        if i_episode % 5 != 0: # more selfplay
            a2 = opponent.select_action(obs_agent2, evaluate=True)
        else:

            
            if opp==-1:
                a2 = basic_strong.act(obs_agent2)
            else:
                a2 = opponents[opp].select_action(obs_agent2,evaluate=True)
        next_state, reward, done, _ = env.step(np.hstack([action[0:4],a2[0:4]])) 
        # env.render()
        
        # env.render()
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if episode_steps == 251 else float(not done)
        # mask = float(not done)
        memory.push(state, action, reward, next_state, mask)

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    games_won = 0
    if i_episode % 13 == 0:
        avg_reward = 0.
        episodes = 11
        for k  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            opp = random.randint(-1, len(opponents)-1)
            while not done:
                
                action = agent.select_action(state, evaluate=True)
                obs_agent2 = env.obs_agent_two()
                if k % 5 != 0: # more selfplay
                    a2 = opponent.select_action(obs_agent2, evaluate=True)
                else:
                    
                    
                    if opp==-1:
                        a2 = basic_strong.act(obs_agent2)
                    else:
                        a2 = opponents[opp].select_action(obs_agent2,evaluate=True)
                next_state, reward, done, info = env.step(np.hstack([action[0:4],a2[0:4]])) 
                if info['winner']==1:
                    games_won+=1
                # env.render()
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        if games_won>6:
            update_now = True
        avg_reward /= episodes
        threshold_avg+=avg_reward

        number_avg+=1

        last_avg = avg_reward

        writer.add_scalar('reward/test', avg_reward, i_episode)
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("Test Episodes: {}, Threshold. Reward: {}".format(episodes, round(threshold_avg/(number_avg), 2)))
        print("----------------------------------------")
    if i_episode%500==0 and (threshold_avg/number_avg)>4.5:
        print('update')
        number_avg=0
        threshold_avg=0
        opponent.policy.load_state_dict(agent.policy.state_dict())
        opponent.critic.load_state_dict(agent.critic.state_dict())
        opponent.critic_target.load_state_dict(agent.critic_target.state_dict())

    if i_episode%500==0:
        threshold_avg=0
        number_avg=0
        time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        agent.save_model( "competionW", "all_win", suffix=f"reward-{last_avg}_episode-"+str(i_episode)+f"_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}_t-{time_}")
    
env.close()


