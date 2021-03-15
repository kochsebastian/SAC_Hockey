import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# hidden hyper params
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """
    Gaussian Policy
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu, log_std

    def sample(self, state):
        '''
        generate sampled action with state as input wrt the policy network
        '''
        mu, log_std = self.forward(state)
        std = log_std.exp() 
        
        normal = Normal(mu, std)
        z = normal.rsample()  # rsample for reparameterization trick 
        # also possible to sample from standard normal and then add mu, times std
        unscaled_action = torch.tanh(z) # TanhNormal distribution as actions
        action = unscaled_action * self.action_scale + self.action_bias # normalize
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. 
        # The TanhNorm forces the Gaussian with infinite action range to be finite. 
        # log probability of action as in common 
        # stochastic Gaussian action policy (without Tanh)
        log_prob = normal.log_prob(z)- torch.log(self.action_scale * (1 - unscaled_action.pow(2)) + epsilon) # epsilon preventing the negative cases in log
        
        # sum up else Multivariate Normal
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias # normalize
        return action, log_prob, mu


    def evaluate(self, state):
        '''
        generate sampled action with state as input wrt the policy network
        '''
        mu, log_std = self.forward(state)
        std = log_std.exp() 
        
        normal = Normal(mu, std)
        z = normal.rsample()  # reparameterization trick 
        # also possible to sample from standard normal and then add mu, times std
        unscaled_action = torch.tanh(z) # TanhNormal distribution as actions
        action = unscaled_action * self.action_scale + self.action_bias # normalize
        
        return action

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)
