
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
parser.add_argument('--env-name', default="Hockey",
                    help='Mujoco Gym environment (default: LunarLanderContinuous-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian (default: Gaussian)')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--wd', type=float, default=0.0, metavar='G',
                    help='learning rate (default: 0.0)')
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
parser.add_argument('--hidden_size', type=int, default=64, metavar='N',
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

actor = "full_player_models/sac_actor_hockey_reward-8.938693169580354_episode-4500_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_23-26-27"
critic = "full_player_models/sac_critic_hockey_reward-8.938693169580354_episode-4500_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-10_23-26-27"
agent.load_model(actor,critic)

args.alpha=0.01
args.automatic_entropy_tuning=False
opponent = SAC(env.observation_space.shape[0], env.action_space, args)
opponent.load_model(actor,critic)
# opponent.al
basic_strong = h_env.BasicOpponent(weak=False)
time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#Tesnorboard
writer = SummaryWriter(f"selfstrongplay-runs/prio{time_}_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}")

# Memory
# memory = PrioritizedReplay(args.replay_size)

memory1 = ReplayMemory(args.replay_size,args.seed)
memory2 = ReplayMemory(args.replay_size,args.seed)

# Training Loop
total_numsteps = 0
updates1 = 0
updates2 = 0


o = env.reset()
# _ = env.render()

for i_episode in itertools.count(1):
    episode_reward1 = 0
    episode_reward2 = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        # state = env.obs_agent_two()
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if args.start_steps > total_numsteps:
            a2 = env.action_space.sample()  # Sample random action
        else:
            a2 = opponent.select_action(state)  # Sample action from policy

        if len(memory1) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss1, critic_2_loss1, policy_loss1, ent_loss1, alpha1, memory_1 = agent.update_parameters(memory1, args.batch_size, updates1)
                memory1=memory_1

                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha1', alpha1, updates1)
                updates1 += 1
        if len(memory2) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss2, critic_2_loss2, policy_loss2, ent_loss2, alpha2, memory_2 = opponent.update_parameters(memory2, args.batch_size, updates2)
                memory2=memory_2

                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha2', alpha2, updates2)
                updates2 += 1

        # a2 = [10,0.,0,0] 
        obs_agent2 = env.obs_agent_two()
        
        next_state, reward, done, info = env.step(np.hstack([action[0:4],a2[0:4]])) 
        reward-=info["reward_closeness_to_puck"]
        
        # env.render()
        episode_steps += 1
        total_numsteps += 1
        episode_reward1 += reward
        episode_reward2 += -reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)

        mask = 1 if episode_steps == 251 else float(not done)
        # mask = float(not done)
        # memory.append(state,action,reward,next_state,mask,episode_done=done)
        # memory.push(state, action, reward, next_state, mask) # Append transition to memory
        memory1.push(state, action, reward, next_state, mask)
        memory2.push(state, action, -reward, next_state, mask)


        state = next_state

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward1: {}, reward2: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward1, 2),round(episode_reward2, 2)))
    # print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward2, 2)))

    if i_episode % 10 == 0:
        avg_reward1 = 0.
        avg_reward2 = 0.

        episodes = 5
        for k  in range(episodes):
            state = env.reset()
            episode_reward1 = 0
            episode_reward2 = 0
            done = False
            while not done:
                
                action = agent.select_action(state, evaluate=True)
                obs_agent2 = env.obs_agent_two()
              
                a2 = opponent.select_action(obs_agent2, evaluate=True)
         
                next_state, reward, done, info = env.step(np.hstack([action[0:4],a2[0:4]]))
                reward-=info["reward_closeness_to_puck"] 
                # env.render()
                episode_reward1 += reward
                episode_reward2 += -reward


                state = next_state
            avg_reward1 += episode_reward1
            avg_reward2 += episode_reward2
        avg_reward1 /= episodes
        avg_reward2 /= episodes

        if i_episode%500==0:
            time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            agent.save_model( "duelplay_models_1", "hockey", suffix=f"reward-{avg_reward1}_episode-"+str(i_episode)+f"_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}_t-{time_}")
            opponent.save_model( "duelplay_models_2", "hockey", suffix=f"reward-{avg_reward2}_episode-"+str(i_episode)+f"_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}_t-{time_}")
       
        writer.add_scalar('avg_reward/agent1', avg_reward1, i_episode)
        writer.add_scalar('avg_reward/agent2', avg_reward2, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward1: {}, Avg. Reward2: {}".format(episodes, round(avg_reward1, 2), round(avg_reward2, 2)))
        print("----------------------------------------")

env.close()


