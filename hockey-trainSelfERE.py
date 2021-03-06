
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
from sac.ere_prio_replay import PrioritizedReplay
from sac.replay_memory import ReplayMemory
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

args.cuda =True if torch.cuda.is_available() else False

env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_model('full_player_models/sac_actor_hockey_11200_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000','full_player_models/sac_critic_hockey_11200_batch_size-4_gamma-0.95_tau-0.005_lr-0.0003_alpha-0.2_tuning-True_hidden_size-256_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000')
opponent = copy.deepcopy(agent)
basic_strong = h_env.BasicOpponent(weak=False)
time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
#Tesnorboard
writer = SummaryWriter(f"selfstrongplay-runs/ere{time_}_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}")

# Memory
memory = PrioritizedReplay(args.replay_size)
# memory = ReplayMemory(args.replay_size,args.seed)

# Training Loop
total_numsteps = 0
updates = 0


o = env.reset()
# _ = env.render()
eta_0 = 0.996
eta_T = 1.0
max_ep_len = 1000 
c_k_min = 2500 # original = 5000
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        # state = env.obs_agent_two()
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy



        # a2 = [10,0.,0,0] 
        obs_agent2 = env.obs_agent_two()
        if i_episode % 4 != 0:
            a2 = opponent.select_action(obs_agent2, evaluate=True)
        else:
            a2 = basic_strong.act(obs_agent2)
        next_state, reward, done, _ = env.step(np.hstack([action[0:4],a2[0:4]])) 
        
        # env.render()
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        mask = 1 if episode_steps == 251 else float(not done)
        # mask = float(not done)

        memory.push(state, action, reward, next_state, mask)
        eta_t = eta_0 + (eta_T - eta_0)*(total_numsteps/args.num_steps)

        state = next_state

    if total_numsteps > args.num_steps:
        break

    for k in range(1,episode_steps):
        c_k = max(int(len(memory)*eta_t**(k*(251/episode_steps))), c_k_min)
        for i in range(args.updates_per_step):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, memory_ = agent.update_parameters(memory, args.batch_size, updates,c_k=c_k)
            memory=memory_

            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temprature/alpha', alpha, updates)
            updates += 1


    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0:
        avg_reward = 0.
        episodes = 5
        for k  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                
                action = agent.select_action(state, evaluate=True)
                obs_agent2 = env.obs_agent_two()
                if k < 4: 
                    a2 = opponent.select_action(obs_agent2, evaluate=True)
                else:
                    a2 = basic_strong.act(obs_agent2)
                next_state, reward, done, _ = env.step(np.hstack([action[0:4],a2[0:4]])) 
                # env.render()
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        if i_episode%500==0:
            time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            agent.save_model( "selfstrongplay_models_ere", "hockey", suffix=f"reward-{avg_reward}_episode-"+str(i_episode)+f"_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}_t-{time_}")
        if i_episode%5000==0:
            opponent.policy.load_state_dict(agent.policy.state_dict())
            opponent.critic.load_state_dict(agent.critic.state_dict())
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()


