import random
import numpy as np
from collections import deque
from prio_replay_memory import PrioritizedReplay as PRE_PrioritizedReplay

# without sum tree
class PrioritizedReplay(PRE_PrioritizedReplay):
    def __init__(self, capacity, alpha=0.6, beta_start = 0.4, beta_steps=100000):
        PRE_PrioritizedReplay.__init__(self, capacity, alpha, beta_start, beta_steps)
        self.buffer     = deque(maxlen=capacity)
        self.prios = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = max(self.prios) if self.buffer else 1.0 
        
        self.buffer.insert(0, (state, action, reward, next_state, done))
        self.prios.insert(0, max_prio)
    
    def sample(self, batch_size, c_k):
        # ere diff
        assert c_k != None
        N = len(self.buffer)
        c_k = max(c_k,N)
  
        prios = np.array(self.prios) if N == self.capacity else np.array(list(self.prios)[:c_k])

        # P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/sum(probs)
        
        # p and the c_k range of the buffer
        indices = np.random.choice(c_k, batch_size, p=P)  # diff to pre
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_step(self.step)
        self.step+=1
                
        # importance-sampling weight
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= max(weights) 
        weights.astype(np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def __len__(self):
        return len(self.buffer)