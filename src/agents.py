from abc import ABC, abstractmethod
from statistics import mean
import tensorflow as tf
import numpy as np

import src.handlers.stats_handler as stats

import src.replay_buffers as buffer_classes
import src.models as model_classes
from src.replay_buffers import ReplayBuffer
from src.models import Model

class Agent(ABC):

    def __init__(self, obs_size, action_size, agn_par, checkpoint_path, eval_mode = False):
       
        self.obs_size = obs_size
        self.action_size = action_size
        self.checkpoint = checkpoint_path
        self.eval_mode = eval_mode

        self.agent_par = agn_par['agn']
        self.train_par = agn_par['trn']

        self.t_step = 0

        self.load_params()
    
    @abstractmethod
    def load_params(self): pass

    @abstractmethod
    def act(self, obs) -> int: pass
        
    @abstractmethod
    def step(self, obs, action, reward, next_obs, done): pass

    @abstractmethod
    def on_episode_start(self): pass

    @abstractmethod
    def on_episode_end(self): pass
    
    @abstractmethod
    def load(self,filename): pass 

    @abstractmethod
    def save(self,filename): pass

    @abstractmethod
    def __str__(self) -> str: pass


class RndAgent(Agent):

    def load_params(self): pass 

    def act(self, obs):
        return np.random.choice(np.arange(self.action_size))

    def step(self, obs, action, reward, next_obs, done): pass

    def on_episode_start(self): pass

    def on_episode_end(self): pass

    def load(self, filename): pass

    def save(self, filename): pass

    def __str__(self):
        return "RandomAgent"


class DQNAgent(Agent):

    def load_params(self):

        self.eps = self.agent_par['eps_start'] if self.eval_mode is False else 0.05
        self.eps_decay = self.agent_par['eps_decay'] 
        self.eps_min = self.agent_par['eps_min']
        
        if not self.eval_mode :
            self.gamma = self.train_par['gamma']
            self.update_every = self.train_par['learn_every']
            self.sample_size = self.train_par['sample_size']
            self.lr = self.train_par['learning_rate']
            
            #Instantiate bufferReplay object 
            buffer_class = getattr(buffer_classes, self.train_par['memory']['class'])
            self.memory : ReplayBuffer = buffer_class(self.train_par['memory']['mem_size'], self.obs_size)   
        
        # Instatiate deep network model 
        if self.checkpoint is not None :
            self.qnetwort = self.load(self.checkpoint)           
        else:
            model_class = getattr(model_classes,self.agent_par['model']['class'])
            self.qnetwork = model_class(self.obs_size,self.action_size,self.lr).get_model()

    def act(self, obs) -> int : 
        state = tf.expand_dims(obs, axis=0)

        if np.random.random() < self.eps: 
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.qnetwork.predict(state))     #TODO change in _local for reusing in sublcass or ovverride 
    

    def step(self, obs, action, reward, next_obs, done):  
        self.t_step +=1

        #store one step experience in replay buffer
        self.memory.store_experience(obs,action,reward,next_obs,done)

        # Learn every UPDATE_EVERY time steps if enough samples are available in memory
        if self.t_step  % self.update_every == 0 and len(self.memory) >= self.sample_size:
            self.learn()


    def learn(self):
        state_sample, action_sample, reward_sample, next_state_sample, done_sample = self.memory.sample_memory(self.sample_size)
    
        q_target = self.qnetwork.predict(state_sample)
        q_next = self.qnetwork.predict(next_state_sample)

        batch_indexes = np.arange(self.sample_size)
    
        q_target[batch_indexes,action_sample] = reward_sample + ((1 - done_sample) * self.gamma * np.max(q_next,axis=1))

        history = self.qnetwork.fit(state_sample, q_target, batch_size = 32, verbose = 0)    

        stats.utils_stats['ep_losses'].append(history.history['loss'][0])
    
    def on_episode_start(self):
        stats.log_stats['eps'] = self.eps
        stats.utils_stats['ep_losses'] = []

    def on_episode_end(self):
        self.eps = max(self.eps_min, self.eps_decay*self.eps)    
        try :
            mean_loss = np.mean(stats.utils_stats['ep_losses'])
            stats.log_stats['mean_episode_loss'] = mean_loss
        except:
             print('Never learned in this episode') 

    def load(self,filename):
        print('loading model from checkpoints/'+filename)
        return tf.keras.models.load_model('checkpoints/'+filename)

    def save(self,filename):
        print('saving model to checkpoints/'+filename)
        self.qnetwork.save('checkpoints/'+filename)  
    
    def __str__(self):
        return 'DQNAgent'


class DoubleDQNAgent(DQNAgent):

    def load_params(self):

        self.eps = self.agent_par['eps_start'] if self.eval_mode is False else 0.05
        self.eps_decay = self.agent_par['eps_decay'] 
        self.eps_min = self.agent_par['eps_min']
        
        if not self.eval_mode :
            self.gamma = self.train_par['gamma']
            self.update_every = self.train_par['learn_every']
            self.sample_size = self.train_par['sample_size']
            self.lr = self.train_par['learning_rate']
            
            #Instantiate bufferReplay object 
            buffer_class = getattr(buffer_classes, self.train_par['memory']['class'])
            self.memory : ReplayBuffer = buffer_class(self.train_par['memory']['mem_size'], self.obs_size)   
        
        # Instatiate deep network model 
        if self.checkpoint is not None :
            self.qnetwork_local = self.load(self.checkpoint+'_local')
            self.qnetwork_target = self.load(self.checkpoint+'_target')   
        else:
            model_class = getattr(model_classes,self.agent_par['model']['class'])
            self.qnetwork_local = model_class(self.obs_size,self.action_size,self.lr).get_model()
            self.qnetwork_target = model_class(self.obs_size,self.action_size,self.lr).get_model()
    
    def learn(self):
        state_sample, action_sample, reward_sample, next_state_sample, done_sample = self.memory.sample_memory(self.sample_size)

        batch_indexes = np.arange(self.sample_size)
    
        q_targets = self.qnetwork_target.predict(state_sample)
        q_next_values = self.qnetwork_target.predict(next_state_sample)[batch_indexes, np.argmax(self.qnetwork_local.predict(next_state_sample), axis=1)]

        q_targets[batch_indexes,action_sample] = reward_sample + ((1 - done_sample) * self.gamma * q_next_values)

        history = self.qnetwork_local.fit(state_sample, q_targets, batch_size = 32, verbose = 0)    

        stats.utils_stats['ep_losses'].append(history.history['loss'][0])

        self.target_update(self)
    
    def target_update(self, tau = 0.5e-3):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_weights, local_weights in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        raise NotImplementedError
        

    def save(self,filename):
        print('saving models to checkpoints/'+filename+'_local'+' and '+filename+'_target')
        self.qnetwork_local.save('checkpoints/'+filename+'_local') 
        self.qnetwork_target.save('checkpoints/'+filename+'_target')   
    
    def __str__(self):
        return 'DoubleDQNAgent'

