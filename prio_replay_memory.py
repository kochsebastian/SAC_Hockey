
import random
import numpy as np

# Without sumtree
class PrioritizedReplay(object):
    def __init__(self, capacity, alpha= 0.6, beta_start = 0.4, beta_steps=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.step = 1 
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.prios = np.zeros((capacity,), dtype=np.float32)
    
    def beta_by_step(self, step_idx):
        # ANNEALING THE BIAS
        return min(1.0, self.beta_start + step_idx * (1.0 - self.beta_start) / self.beta_steps)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = self.prios.max() if self.buffer else 1.0 # no td error available when push
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            # buffer full, replace oldest mem
            self.buffer[self.pos] = (state, action, reward, next_state, done) 
        
        self.prios[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity 
    
    def sample(self, batch_size, c_k=None):
        assert c_k == None

        N = len(self.buffer)
        prios = self.prios if N == self.capacity else self.prios[:self.pos]
            
        # P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/sum(probs)
        
        # choose by prob
        indices = np.random.choice(N, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_step(self.step)
        self.step+=1
                
        # importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        
        weights /= max(weights)
        weights.astype(np.float32)
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.prios[idx] = abs(prio) 

    def __len__(self):
        return len(self.buffer)
