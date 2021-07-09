from abc import ABC, abstractmethod
from typing import Deque
from collections import namedtuple, deque
from collections.abc import Iterable
import random
import numpy as np 

class ReplayBuffer(ABC):

    def __init__(self,mem_size, obs_size = None):
        self.mem_size = mem_size
        self.obs_size = obs_size
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

    def __init__(self, mem_size, obs_size):
        super().__init__(mem_size,obs_size)

        self.state_memory = np.zeros((mem_size,obs_size))
        self.action_memory = np.zeros((mem_size),dtype=np.int8)
        self.reward_memory = np.zeros((mem_size))
        self.new_state_memory = np.zeros((mem_size,obs_size))
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

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample
    
    def __str__(self):
        return 'ReplayBuffer_np'


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer_dq(ReplayBuffer):

    def __init__(self, mem_size, obs_size):
        self.memory = deque(maxlen=mem_size)
        super().__init__(mem_size,obs_size)
        
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

        return state_sample, action_sample, reward_sample, new_state_sample, done_sample

    def __v_stack_impr(self, states):
        np_states = np.array(states)
        if isinstance(states[0], Iterable):
            np_states = np.reshape(np_states, (len(states), len(states[0])))
        return np_states

    def __str__(self):
        return 'ReplayBuffer_dq'


class PPOAgentBuffer:

    def init(self):
        self.memory = {}
        self.stored = 0


    def store_agent_experience(self, agent_id, action, value, obs, reward, done, policy_logits):
        experience = self.memory.get(agent_id, [])
        experience.append([action, value, obs, reward, done, policy_logits])
        self.memory[agent_id] = experience
        self.stored +=1 

    def get_agent_experience(self, agent_id):
        action, value, obs, reward, done, policy_logits = [np.squeeze(i) for i in zip(*self.memory[agent_id])]
        return action, value, obs, reward, done, policy_logits

    def reset_mem(self):
        self.memory = {}

    def len(self):
        return self.stored

    def str(self):
        return 'PPOAgentBuffer'