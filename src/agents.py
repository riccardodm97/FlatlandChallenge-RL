from tensorflow.python.training.tracking.util import Checkpoint
from src.replay_buffers import ReplayBuffer_np
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self, obs_size, action_size, agent_par, train_par, checkpoint_path):
       
        self.obs_size = obs_size
        self.action_size = action_size
        self.checkpoint = checkpoint_path

        self.agent_par = agent_par
        self.train_par = train_par

        self.t_step = 0

    @abstractmethod
    def act(self, obs) -> int: pass
        
    @abstractmethod
    def step(self, obs, action, reward, next_obs, done): pass

    @abstractmethod
    def learn(self): pass

    @abstractmethod
    def on_episode_end(self): pass
        
    @abstractmethod
    def __str__(self): pass
        
    def load_model(self):
        return tf.keras.models.load_model(self.checkpoint)

    def save_model(self):
        self.qnetwork.save()  #TODO implement 


class DQNAgent(Agent):

    def __init__(self, obs_size, action_size, agent_par, train_par, checkpoint_path):

        super().__init__(obs_size,action_size,agent_par,train_par,checkpoint_path)

        self.eps = self.agent_par['eps_start'] 
        self.eps_decay = self.agent_par['eps_decay']
        self.eps_min = self.agent_par['eps_min']
        
        self.gamma = self.train_par['gamma']
        self.update_every = self.train_par['learn_every']
        self.sample_size = self.train_par['sample_size']
        self.mem_size = self.train_par['mem_size']
        self.lr = self.train_par['learning_rate']
        
        self.memory = ReplayBuffer_np(self.mem_size,self.obs_size)
        self.init_qnetwork()

    def init_qnetwork(self):

        model = None
        if self.checkpoint is not None :
            model = self.load_model(self.checkpoint)
        else:
            model = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(128, input_shape=(self.obs_size,)),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.Activation('relu'),
                    tf.keras.layers.Dense(self.action_size)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr), loss='mse')

        self.qnetwork = model

    def on_episode_end(self):
        self.eps = max(self.eps_min, self.eps_decay*self.eps)      #TODO evitare che debbano averlo tutti gli agent

    def act(self, obs) -> int : 
        state = tf.expand_dims(obs, axis=0)
        if np.random.random() <= self.eps: 
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.qnetwork.predict(state))
    

    def step(self, obs, action, reward, next_obs, done):
        
        self.t_step +=1

        #store one step experience in replay buffer
        self.memory.store_experience(obs,action,reward,next_obs,done)

        # Learn every UPDATE_EVERY time steps if enough samples are available in memory
        if self.t_step  % self.update_every == 0 and self.memory.stored >= self.sample_size:
            self.learn()


    def learn(self):
        state_sample, action_sample, reward_sample, next_state_sample, done_sample = self.memory.sample_memory(self.sample_size)
        
        q_next = self.qnetwork.predict(next_state_sample)
        q_target = self.qnetwork.predict(state_sample)

        batch_indexes = np.arange(self.sample_size)
    
        q_target[batch_indexes,action_sample] = reward_sample + (self.gamma * np.max(q_next,axis=1) * (1 - done_sample))

        self.qnetwork.fit(state_sample,q_target,batch_size = 32,verbose = 0)    
    
    def __str__(self):
        return 'DQNAgent'

