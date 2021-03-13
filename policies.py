import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
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
        generate sampled action with state as input wrt the policy network;
        '''
        mu, log_std = self.forward(state)
        std = log_std.exp() 
        normal = Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mu + std * N(0,1))
        y_t = torch.tanh(x_t) # TanhNormal distribution as actions
        action = y_t * self.action_scale + self.action_bias
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        log_prob = normal.log_prob(x_t)
        # caused by the Tanh(), as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias
        return action, log_prob, mu

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)
