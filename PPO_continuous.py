import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
import gym
import optparse
import pickle
import laserhockey.hockey_env as h_env
from torch.utils.tensorboard import SummaryWriter
import itertools
import datetime


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, n_latent_var=64):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Tanh()
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )
        
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory,eval=False):
        state = torch.from_numpy(state).float().to(device)
        action_mean = self.action_layer(state)	
        cov_mat = torch.diag(self.action_var).float().to(device)	
        	
        dist = MultivariateNormal(action_mean, cov_mat)	
        action = dist.sample()	
        action_logprob = dist.log_prob(action)

        # state = torch.from_numpy(state).float().to(device)
        # action_probs = self.action_layer(state)
        # dist = Categorical(action_probs)
        # action = dist.sample() 
        
        if not eval:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)

        return action.detach().cpu().numpy()

    def evaluate(self, state, action):
        action_mean = self.action_layer(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

        

class PPO:
    def __init__(self, state_dim, action_dim, action_std, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):

            # Evaluating old actions and values: use policy.evaluate
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Hints:
            #  for the ratio (pi_theta / pi_theta__old), note that you have log probabilities given
            #  compute the advantage using the Monte-Carlo Advantage Estimator
            #  you don't want to backpropagate through the values here, so use detach()
            #  compute the two objectives, normal and clipped
            ratios = torch.exp(logprobs - old_logprobs.detach()) # real ratio
            advantages = rewards - state_values.detach() # expected rewards - baseline (reduces variance)
            objective = ratios * advantages
            objective_clipped = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # --- the rest is given
            # the loss is given for you with the magic constants for the value function term and the policy entropy term
            loss = -torch.min(objective, objective_clipped) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())



optParser = optparse.OptionParser()
optParser.add_option('-e', '--env',action='store', type='string',
                        dest='env_name',default="hockey",
                        help='Environment (default %default)')
optParser.add_option('-c', '--eps',action='store',  type='float',
                        dest='eps_clip',default=0.2,
                        help='Clipping epsilon (default %default)')
optParser.add_option('-r', '--run',action='store',  type='int',
                        dest='test_run',default=0,
                        help='Test run (default %default)')



env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

opts, args = optParser.parse_args()
############## Hyperparameters ##############
run_number = opts.test_run
env_name = opts.env_name
# creating environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
render = False
solved_reward = 230         # stop training if avg_reward > solved_reward
log_interval = 20           # print avg reward in the interval
max_interactions = 10000001        # max training episodes
max_timesteps = 300         # max timesteps in one episode
n_latent_var = 256           # number of variables in hidden layer
update_timestep = 2000      # update policy every n timesteps
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.95                # discount factor
action_std = 0.5
K_epochs = 10               # update policy for K epochs
eps_clip = opts.eps_clip    # clip parameter for PPO
random_seed = None
#############################################


if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory = Memory()
ppo = PPO(state_dim, action_dim, action_std, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
print(env_name,"Clipping:", eps_clip)
time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

writer = SummaryWriter(f"ppo_baseline/{time_}__gamma-{gamma}_lr-{lr}_hidden_size-{n_latent_var}")

# logging variables
rewards = []
lengths = []
timestep = 0
total_numsteps = 0


o = env.reset()



for i_episode in itertools.count(1):
    running_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    for t in range(max_timesteps):
        timestep += 1
        a1 = ppo.policy_old.act(state, memory)
        # a1 = np.random.uniform(-1, 1, 4)
        a2 = np.array([10.,0.,0.,0.])
        next_state, reward, done, info = env.step(np.hstack([a1[0:4],a2[0:4]])) 
        # env.render()

        # Saving reward and is_terminal:
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        if len(memory.states)!=len(memory.rewards):
            print()
        # update if its time
        if timestep % update_timestep == 0:
            ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        running_reward += reward
        if render:
            env.render()
        if done:
            break


        state = next_state

        episode_steps += 1
        total_numsteps += 1

    if total_numsteps > max_interactions:
        break


    writer.add_scalar('reward/train', running_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(running_reward, 2)))

    if i_episode % 10 == 0 :
        avg_reward = 0.
        episodes = 5
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                
                a1 = ppo.policy_old.act(state, memory,eval=True)
                # a1 = np.random.rand(4)
                a2 = np.array([10.,0.,0.,0.])

                next_state, reward, done, info = env.step(np.hstack([a1[0:4],a2[0:4]]))  
                # env.render()
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        # if i_episode%100==0:
        #     time_ = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        #     agent.save_model( "hockey-ere-models-attack", "hockey", suffix=f"reward-{avg_reward}_episode-"+str(i_episode)+f"_batch_size-{args.batch_size}_gamma-{args.gamma}_tau-{args.tau}_lr-{args.lr}_alpha-{args.alpha}_tuning-{args.automatic_entropy_tuning}_hidden_size-{args.hidden_size}_updatesStep-{args.updates_per_step}_startSteps-{args.start_steps}_targetIntervall-{args.target_update_interval}_replaysize-{args.replay_size}_t-{time_}")
  
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()


