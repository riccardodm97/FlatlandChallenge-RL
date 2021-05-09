import numpy as np 
from abc import ABC, abstractmethod
from collections import namedtuple, deque, Iterable
import random
import tensorflow as tf 


class ReplayBuffer(ABC):

    def __init__(self,mem_size):
        self.mem_size = mem_size
        self.stored = 0

    @abstractmethod
    def store_experience(self,state,action,reward,new_state,done):
        pass

    @abstractmethod
    def sample_memory(self,sample_size):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __len__(self):
        return self.stored
    

class ReplayBuffer_np(ReplayBuffer):

    def __init__(self, mem_size, input_shape):
        self.state_memory = np.zeros((mem_size,input_shape))
        self.action_memory = np.zeros((mem_size),dtype=np.int8)
        self.reward_memory = np.zeros((mem_size))
        self.new_state_memory = np.zeros((mem_size,input_shape))
        self.done_memory = np.zeros((mem_size),dtype=np.int8)
        super().__init__(mem_size)
    
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

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample
    
    def __str__(self):
        return 'ReplayBuffer_np'


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer_dq(ReplayBuffer):

    def __init__(self, mem_size):
        self.memory = deque(maxlen=mem_size)
        super().__init__(mem_size)
        
    def store_experience(self, state, action, reward, new_state, done):
        e = Experience(np.expand_dims(state, 0), action, reward, np.expand_dims(new_state, 0), int(done))
        self.memory.append(e)
        self.stored += 1

    def sample_memory(self,sample_size):
        experiences = random.sample(self.memory, k=sample_size)
        state_sample = self.__v_stack_impr([e.state for e in experiences if e is not None])
        action_sample = self.__v_stack_impr([e.action for e in experiences if e is not None])
        reward_sample = self.__v_stack_impr([e.reward for e in experiences if e is not None])
        new_state_sample = self.__v_stack_impr([e.next_state for e in experiences if e is not None])
        done_sample = self.__v_stack_impr([e.done for e in experiences if e is not None])

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample

    def __v_stack_impr(self, states):
        sub_dim = len(states[0][0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states

    def __str__(self):
        return 'ReplayBuffer_dq'
