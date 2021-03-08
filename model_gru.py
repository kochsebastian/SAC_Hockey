import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(SoftQNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.gru3 = nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.linear4 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear6 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear7 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.gru8 = nn.GRU(hidden_dim,hidden_dim,batch_first=True)
        self.linear9 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear10 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, last_action, hidden_in):
        # state shape: (batch_size, sequence_length, state_dim)
        # rnn needs:  (sequence_length, batch_size, state_dim) 
        # state = state.permute(1,0,2) 
        # action = action.permute(1,0,2)
        # last_action = last_action.permute(1,0,2)

        xu = torch.cat([state, action], -1)
        xv = torch.cat([state, last_action], -1)
        
        fc1 = F.relu(self.linear1(xu))
        rnn1 = F.relu(self.linear2(xv))
        rnn1, rnn_hidden = self.gru3(rnn1, hidden_in) # non linearity necessary?
        # rnn1 = F.relu(rnn1)
        merge1 = torch.cat([fc1, rnn1], -1) 

        x1 = F.relu(self.linear4(merge1))
        x1 = self.linear5(x1)

   
        fc2 = F.relu(self.linear6(xu))
        rnn2 = F.relu(self.linear7(xv))
        rnn2, rnn_hidden = self.gru8(rnn2, hidden_in) # non linearity necessary?
        # rnn2 = F.relu(rnn2)
        merge2 = torch.cat([fc2, rnn2], -1) 

        x2 = F.relu(self.linear9(merge2))
        x2 = self.linear10(x2)

        # x1 = x1.permute(1,0,2)  # back to same axes as input  
        # x2 = x2.permute(1,0,2)  # back to same axes as input 
        return x1, x2


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(PolicyNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(num_inputs+num_actions, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)

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

    def forward(self, state, last_action, hidden_in):
        # state shape: (batch_size, sequence_length, state_dim)
        # rnn needs:  (sequence_length, batch_size, state_im) 
        # state = state.permute(1,0,2)
        # last_action = last_action.permute(1,0,2)
        xu =  torch.cat([state, last_action], -1)

        fc = F.relu(self.linear1(state))
        rnn = F.relu(self.linear2(xu))
        rnn, rnn_hidden = self.gru(rnn, hidden_in) # non linearity necessary?
        # rnn = F.relu(rnn)
        merge = torch.cat([fc, rnn], -1) 
        x = F.relu(self.linear3(merge))
        # x = x.permute(1,0,2)  # permute back

        mean = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std, rnn_hidden

    def sample(self, state, last_action, hidden_in):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp() # no clip in sampling, clip affects gradients flow
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
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
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, hidden_out

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(PolicyNetwork, self).to(device)


# class DeterministicPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(DeterministicPolicy, self).__init__()
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)

#         self.mean = nn.Linear(hidden_dim, num_actions)
#         self.noise = torch.Tensor(num_actions)

#         self.apply(weights_init_)

#         # action rescaling
#         if action_space is None:
#             self.action_scale = 1.
#             self.action_bias = 0.
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
#         return mean

#     def sample(self, state):
#         mean = self.forward(state)
#         noise = self.noise.normal_(0., std=0.1)
#         noise = noise.clamp(-0.25, 0.25)
#         action = mean + noise
#         return action, torch.tensor(0.), mean

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         self.noise = self.noise.to(device)
#         return super(DeterministicPolicy, self).to(device)
