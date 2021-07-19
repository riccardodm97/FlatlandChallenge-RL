from abc import ABC, abstractmethod
from typing import Deque
from collections import namedtuple, deque
from collections.abc import Iterable
import random
import numpy as np 

from src.utils.sumTree import SumTree

class ReplayBuffer(ABC):

    def __init__(self,mem_size,obs_shape = None):
        self.mem_size = mem_size
        self.obs_size = np.prod(obs_shape)
        self.stored = 0

    @abstractmethod
    def store_experience(self,state,action,reward,new_state,done):
        pass

    @abstractmethod
    def sample_memory(self,sample_size):
        pass
    
    @abstractmethod
    def buffer_update(self,abs_errors):
        pass

    @abstractmethod
    def __str__(self):
        pass
    
    def __len__(self):
        return self.stored
    

class ReplayBuffer_np(ReplayBuffer):

    def __init__(self, mem_size, obs_shape):
        super().__init__(mem_size,obs_shape)

        self.state_memory = np.zeros((mem_size,self.obs_size))
        self.action_memory = np.zeros((mem_size),dtype=np.int8)
        self.reward_memory = np.zeros((mem_size))
        self.new_state_memory = np.zeros((mem_size,self.obs_size))
        self.done_memory = np.zeros((mem_size),dtype=np.int8)
        
    
    def store_experience(self, state, action, reward, new_state, done):
        index = self.stored % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.done_memory[index] = int(done)
        self.stored += 1

    def sample_memory(self, sample_size):
        sample_indices = np.random.choice(min(self.mem_size,self.stored),size=sample_size)
        state_sample = self.state_memory[sample_indices]
        action_sample = self.action_memory[sample_indices]
        reward_sample = self.reward_memory[sample_indices]
        new_state_sample = self.new_state_memory[sample_indices]
        done_sample = self.done_memory[sample_indices] 

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample, None
    
    def buffer_update(self,abs_errors):
        pass
    
    def __str__(self):
        return 'ReplayBuffer_np'


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer_dq(ReplayBuffer):

    def __init__(self, mem_size, obs_shape):
        self.memory = deque(maxlen=mem_size)
        super().__init__(mem_size,obs_shape)
        
    def store_experience(self, state, action, reward, new_state, done):
        e = Experience(np.array(state), action, reward, np.array(new_state), int(done))
        self.memory.append(e)
        self.stored += 1

    def sample_memory(self,sample_size):
        experiences : Deque[Experience] = random.sample(self.memory, k=sample_size)
        state_sample = self.__v_stack_impr([e.state for e in experiences if e is not None])
        action_sample = self.__v_stack_impr([e.action for e in experiences if e is not None])
        reward_sample = self.__v_stack_impr([e.reward for e in experiences if e is not None])
        new_state_sample = self.__v_stack_impr([e.next_state for e in experiences if e is not None])
        done_sample = self.__v_stack_impr([e.done for e in experiences if e is not None])

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample, None
    
    def buffer_update(self,abs_errors):
        pass

    def __v_stack_impr(self, states):
        np_states = np.array(states)
        if isinstance(states[0], Iterable):
            np_states = np.reshape(np_states, (len(states), len(states[0])))
        return np_states

    def __str__(self):
        return 'ReplayBuffer_dq'


class PPOAgentBuffer:

    def __init__(self):
        self.memory = {}
        self.stored = 0

    def store_agent_experience(self, agent_id, action, value, obs, reward, done, policy_logits):
        experience = self.memory.get(agent_id, [])
        experience.append([action, value, obs, reward, int(done), policy_logits])
        self.memory[agent_id] = experience
        self.stored +=1 

    def get_agent_experience(self, agent_id):
        action, value, obs, reward, done, policy_logits = [np.squeeze(i) for i in zip(*self.memory[agent_id])]
        return action, value, obs, reward, done, policy_logits

    def reset_mem(self):
        self.memory = {}

    def __len__(self):
        return self.stored

    def __str__(self):
        return 'PPOAgentBuffer'


class PrioritizedReplayBuffer(ReplayBuffer):  
   
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # Importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, mem_size, obs_shape):
        super().__init__(mem_size,obs_shape)

        self.tree = SumTree(self.mem_size)

    
    def store_experience(self,state,action,reward,new_state,done):
        
        experience = state, action, reward, new_state, int(done)

        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max p for new p

        self.stored += 1


    def sample_memory(self, sample_size):
        # Create a sample array that will contains the minibatch
        memory = []
        idxs = []
        priorities = []

        # Compute the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / sample_size       # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1


        for i in range(sample_size):
            
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            
            idxs.append(index)
            priorities.append(priority)
            memory.append(data)

        
        sampling_probabilities = priorities / self.tree.total_priority

        isWeights = np.power(self.tree.n_entries* sampling_probabilities, -self.PER_b)      #IsWeights = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
        isWeights /= isWeights.max()

        self.last_sample_idxs = idxs                      

        state_sample, action_sample, reward_sample, next_state_sample, done_sample = [np.squeeze(i) for i in zip(*memory)]

        return state_sample, action_sample, reward_sample, next_state_sample, done_sample, isWeights

    #Update the priorities on the tree
    def buffer_update(self, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        tree_idx = self.last_sample_idxs
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
    def __str__(self):
        return 'PrioritizedReplayBuffer'