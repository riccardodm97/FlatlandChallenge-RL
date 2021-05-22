from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf
from tensorflow import keras

import src.utils.stats_handler as stats

import src.policy.replay_buffers as buffer_classes
import src.policy.models as model_classes
from src.policy.replay_buffers import ReplayBuffer

class Agent(ABC):

    def __init__(self, obs_size, action_size, agent_par, checkpoint_path, eval_mode = False):
       
        self.obs_size = obs_size
        self.action_size = action_size
        self.checkpoint = checkpoint_path
        self.eval_mode = eval_mode

        self.agent_par = agent_par

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

        if not self.agent_par['noisy']:

            self.eps = self.agent_par['eps_start'] if self.eval_mode is False else 0.05
            self.eps_decay = self.agent_par['eps_decay'] 
            self.eps_min = self.agent_par['eps_min']
        
        if not self.eval_mode :
            self.gamma = self.agent_par['gamma']
            self.update_every = self.agent_par['learn_every']
            self.sample_size = self.agent_par['sample_size']
            self.lr = self.agent_par['learning_rate']
            
            #Instantiate bufferReplay object 
            buffer_class = getattr(buffer_classes, self.agent_par['memory']['class'])
            self.memory : ReplayBuffer = buffer_class(self.agent_par['memory']['mem_size'], self.obs_size)   
        
        # Instatiate deep network model 
        if self.checkpoint is not None :
            self.qnetwort = self.load(self.checkpoint)     
        else:
            model_class = getattr(model_classes,self.agent_par['model_class'])
            self.qnetwork = model_class(self.obs_size,self.action_size,self.lr).get_compiled_model()
            
        #if double qNetwork instantiate target-model also 
        if self.agent_par['double']:
            model_class = getattr(model_classes,self.agent_par['model_class'])
            self.qnetwork_target = model_class(self.obs_size,self.action_size,self.lr).get_model()
            self.qnetwork_target.set_weights(self.qnetwork.get_weights())  
         

    def act(self, obs) -> int : 
        state = tf.expand_dims(obs, axis=0)

        if np.random.random() < self.eps and not self.agent_par['noisy']: 
            stats.log_stats['random_action_taken'] += 1
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.qnetwork.predict(state))     
    

    def step(self, obs, action, reward, next_obs, done):  
        self.t_step +=1

        #store one step experience in replay buffer
        self.memory.store_experience(obs,action,reward,next_obs,done)

        # Learn every UPDATE_EVERY time steps if enough samples are available in memory
        if self.t_step  % self.update_every == 0 and len(self.memory) >= self.sample_size:
            self.learn()

    def learn(self):
        state_sample, action_sample, reward_sample, next_state_sample, done_sample = self.memory.sample_memory(self.sample_size)

        batch_indexes = np.arange(self.sample_size)

        q_targets = self.qnetwork.predict(state_sample)

        if self.agent_par['double']:
            q_next_values = self.qnetwork_target.predict(next_state_sample)[batch_indexes, np.argmax(self.qnetwork.predict(next_state_sample), axis=1)]
        else:
            q_next_values = np.max(self.qnetwork.predict(next_state_sample),axis=1)

        q_targets[batch_indexes,action_sample] = reward_sample + ((1 - done_sample) * self.gamma * q_next_values)

        history = self.qnetwork.fit(state_sample, q_targets, batch_size = 32, verbose = 0)    

        stats.utils_stats['ep_losses'].append(history.history['loss'][0])

        if self.agent_par['double']:
            self.target_update(self.agent_par['tau'])
    
    def target_update(self, tau = 0.5e-3):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        #self.qnetwork_target.set_weights(tau * np.array(self.qnetwork.get_weights()) + (1.0 - tau) * np.array(self.qnetwork_target.get_weights())) #TODO verificare
        for t, e in zip(self.qnetwork_target.trainable_variables, self.qnetwork.trainable_variables):
            t.assign(t * (1.0 - tau) + e * tau)

    
    def on_episode_start(self):
        stats.log_stats['eps'] = self.eps
        stats.utils_stats['ep_losses'] = []
        stats.log_stats['random_action_taken'] = 0

    def on_episode_end(self):
        self.eps = max(self.eps_min, self.eps_decay*self.eps)    
        try :
            mean_loss = np.mean(stats.utils_stats['ep_losses'])
            std_loss = np.std(stats.utils_stats['ep_losses'])
            stats.log_stats['mean_episode_loss'] = mean_loss
            stats.log_stats['std_episode_loss'] = std_loss
        except:
             print('Never learned in this episode') 
            

    def load(self,filename):
        print('loading model from checkpoints/'+filename)
        return keras.models.load_model('checkpoints/'+filename)

    def save(self,filename):
        print('saving model to checkpoints/'+filename)
        self.qnetwork.save('checkpoints/'+filename)  
    
    def __str__(self):
        return 'DQNAgent'


