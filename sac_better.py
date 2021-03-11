import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import PolicyNetwork, SoftQNetwork
from ere_prio_replay import PrioritizedReplay as ERE_PrioritizedReplay
from prio_replay_memory import PrioritizedReplay

class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = SoftQNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = SoftQNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = PolicyNetwork(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            # gSDE? noisy layers?
            raise NotImplementedError

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, log_prob, mu = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def deputy_mse(self,qf1,qf2,next_q_value,weights):
        # to incorperate the weights for pre
        td_error1 =  next_q_value - qf1
        td_error2 =  next_q_value - qf2
        qf1_loss = 0.5 * (td_error1.pow(2)*weights).mean()
        qf2_loss = 0.5 * (td_error2.pow(2)*weights).mean()
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())
        return qf1_loss, qf2_loss, prios

    def update_parameters(self, memory, batch_size, updates, c_k=None):
        # Sample a batch from memory
        if isinstance(memory,PrioritizedReplay) or isinstance(memory,ERE_PrioritizedReplay):
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, idxs, weights_batch = memory.sample(batch_size=batch_size,c_k=c_k)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        if isinstance(memory,PrioritizedReplay) or isinstance(memory,ERE_PrioritizedReplay):
            weights_batch = torch.FloatTensor(weights_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch) 
        if isinstance(memory,PrioritizedReplay) or isinstance(memory,ERE_PrioritizedReplay):
            qf1_loss, qf2_loss, prios = self.deputy_mse(qf1, qf2, next_q_value, weights_batch)
        else:
            qf1_loss = F.mse_loss(qf1, next_q_value)  
            qf2_loss = F.mse_loss(qf2, next_q_value)  
        qf_loss = qf1_loss + qf2_loss
        

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if isinstance(memory,PrioritizedReplay) or isinstance(memory,ERE_PrioritizedReplay):
            memory.update_priorities(idxs, prios.data.cpu().numpy())

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        if isinstance(memory,PrioritizedReplay) or isinstance(memory,ERE_PrioritizedReplay):
            policy_loss = ((self.alpha * log_pi) - min_qf_pi*weights_batch).mean() 
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() 
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) 


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), memory

    # Save model parameters
    def save_model(self, root_name, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists(root_name):
            os.makedirs(root_name)

        if actor_path is None:
            actor_path = "{}/sac_actor_{}_{}".format(root_name,env_name, suffix)
        if critic_path is None:
            critic_path = "{}/sac_critic_{}_{}".format(root_name,env_name, suffix)
        print('Saving models to {} and {}'.format(root_name,actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

