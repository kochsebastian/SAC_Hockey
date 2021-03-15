import numpy as np

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client
import argparse
import torch
from sac_better import SAC
import laserhockey.hockey_env as h_env

import gym

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


class SACAgent(BasicOpponent, RemoteControllerInterface):

    def __init__(self, weak, keep_mode=True):
        self.agent = SAC(env.observation_space.shape[0], env.action_space, args)
        root = "/home/sebastiankoch/SoftActorCriticRNN/finals/alpha/"
        actor = root+"sac_actor_500updates_hockey_reward--0.20085748776545811_episode-33000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.02_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_06-01-44"
        critic = root+"sac_critic_500updates_hockey_reward--0.20085748776545811_episode-33000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.02_tuning-False_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-5_replaysize-10000000_t-2021-03-14_06-01-44"
        self.agent.load_model(actor,critic,None)
        # root = "/home/sebastiankoch/SoftActorCriticRNN/finals/new_best/"
        # actor = root+"sac_actor_500updates_win_reward-7.166574252130548_episode-5000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-15_14-49-13"
        # critic = root+"sac_critic_500updates_win_reward-7.166574252130548_episode-5000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-15_14-49-13"
        # target = root+"sac_target_500updates_win_reward-7.166574252130548_episode-5000_batch_size-8_gamma-0.97_tau-0.005_lr-0.0003_alpha-0.01_tuning-True_hidden_size-512_updatesStep-1_startSteps-10000_targetIntervall-1_replaysize-1000000_t-2021-03-15_14-49-13"
        # self.agent.load_model(actor,critic,target)
        RemoteControllerInterface.__init__(self, identifier='SAC')

    def remote_act(self, 
            obs : np.ndarray,
           ) -> np.ndarray:

        return self.agent.select_action(obs,evaluate=True)
        

if __name__ == '__main__':
    controller = SACAgent(weak=False)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='Sebastian_Koch_Kexbot', # Testuser
                    password='vN.g!5Fd',
                    controller=controller, 
                    output_path='/tmp/ALRL2020/client/Sebastian_Koch_Kexbot', # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)
