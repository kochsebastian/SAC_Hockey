import random
import numpy as np
from collections import deque
from prio_replay_memory import PrioritizedReplay

# without sum tree
class PrioritizedReplay(PrioritizedReplay):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, alpha=0.6, beta_start = 0.4, beta_steps=100000):
        PrioritizedReplay.__init__(self,capacity, alpha, beta_start, beta_steps)
        self.buffer     = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = max(self.priorities) if self.buffer else 1.0 
        
        self.buffer.insert(0, (state, action, reward, next_state, done))
        self.priorities.insert(0, max_prio)
    
    
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k > N:
            c_k = N
        
        if N == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:c_k])
        
        #(prios)
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_step(self.step)
        self.step+=1
                
        #Compute importance-sampling weight
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def __len__(self):
        return len(self.buffer)