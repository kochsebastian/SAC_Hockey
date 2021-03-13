import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from policies import Actor
from softqnetwork import Critic
import numpy as np

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = Critic(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)


        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = Actor(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            raise NotImplementedError
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self,target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state, last_action, hidden_in, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0)
        last_action = torch.FloatTensor(last_action).to(self.device).unsqueeze(0).unsqueeze(0)
        if evaluate is False:
            action, _, _, hidden_out = self.policy.sample(state, last_action, hidden_in)
        else:
            _, _, action, hidden_out = self.policy.sample(state, last_action, hidden_in)
        return action.detach().cpu().numpy()[0][0], hidden_out

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        hidden_in_batch, hidden_out_batch, state_batch, action_batch, last_action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        pad_batch = max(len(l) for l in state_batch)
        pad_state = np.zeros(state_batch[0][0].shape, dtype="float32")
        pad_action = np.zeros(action_batch[0][0].shape, dtype="float32")
        pad_reward = state_batch[]
        pad_done = 1.0
        for i,s in enumerate(state_batch):
            l = len(s)
            if pad_batch-l==0:
                continue
            state_batch[i].extend(([pad_state]*(pad_batch-l)) )
            next_state_batch[i].extend(([pad_state]*(pad_batch-l)))
            last_action_batch[i].append(action_batch[i][-1])
            action_batch[i].extend(([pad_action]*(pad_batch-l))) 
            last_action_batch[i].extend(([pad_action]*(pad_batch-l-1)) )
            reward_batch[i].extend(([pad_reward]*(pad_batch-l)) )
            mask_batch[i].extend(([pad_done]*(pad_batch-l))) 

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        last_action_batch = torch.FloatTensor(last_action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(-1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(-1)

        

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state_batch, last_action_batch, hidden_in_batch)
            new_next_state_action, new_next_state_log_pi, _, _ = self.policy.sample(next_state_batch, action_batch, hidden_out_batch)
           
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, new_next_state_action, action_batch, hidden_out_batch)
            
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch, last_action_batch, hidden_in_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # attention here I have dim problems
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, next_state_action, last_action_batch, hidden_in_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * next_state_log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (next_state_log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            if self.tau == 1:
                self.hard_update(self.critic_target,self.critic)
            else:
                self.soft_update(self.critic_target, self.critic, self.tau)
        

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

