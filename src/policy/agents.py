from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf
from tensorflow import keras

import src.utils.stats_handler as stats
import src.policy.replay_buffers as buffer_classes
import src.policy.models as model_classes
import src.policy.action_selectors as action_sel_classes

from src.policy.replay_buffers import PPOAgentBuffer, PrioritizedReplayBuffer, ReplayBuffer
from src.policy.action_selectors import ActionSelector,GreedyAS

from tensorflow.keras.optimizers import Adam

class Agent(ABC):

    def __init__(self, obs_shape, action_size, agent_par, checkpoint_path, eval_mode = False):
       
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.checkpoint = checkpoint_path
        self.eval_mode = eval_mode

        self.agent_par = agent_par

        self.t_step = 0

        self.load_params()
    
    @abstractmethod
    def load_params(self): pass

    @abstractmethod
    def act(self, obs, eval_mode) -> int: pass
        
    @abstractmethod
    def step(self, obs, action, reward, next_obs, done, agent): pass

    @abstractmethod
    def on_episode_start(self): pass

    @abstractmethod
    def on_episode_end(self, agents): pass
    
    @abstractmethod
    def load(self,filename): pass 

    @abstractmethod
    def save(self,filename): pass

    @abstractmethod
    def __str__(self) -> str: pass


class RndAgent(Agent):

    def load_params(self): pass 

    def act(self, obs, eval_mode):
        return np.random.choice(np.arange(self.action_size))        #TODO use RandomAS

    def step(self, obs, action, reward, next_obs, done, agent): pass

    def on_episode_start(self): pass

    def on_episode_end(self, agents): pass

    def load(self, filename): pass

    def save(self, filename): pass

    def __str__(self):
        return "RandomAgent"


class DQNAgent(Agent):

    def load_params(self):
        
        if not self.eval_mode :
            self.gamma = self.agent_par['gamma']
            self.update_every = self.agent_par['learn_every']
            self.sample_size = self.agent_par['sample_size']
            self.lr = self.agent_par['learning_rate']
            
            #Instantiate bufferReplay object 
            buffer_class = getattr(buffer_classes, self.agent_par['memory']['class'])
            self.memory : ReplayBuffer = buffer_class(self.agent_par['memory']['mem_size'], self.obs_shape) 
            self.mem_is_PER = self.agent_par['memory']['is_per']
            if self.mem_is_PER:
                assert isinstance(self.memory,PrioritizedReplayBuffer)     #if mem_is_PER is true the buffer SHOULD be a PrioritizedExperienceReplay 

        #Instantiate action selector
        action_sel_class = getattr(action_sel_classes, self.agent_par['action_selection']['class'])
        self.action_selector : ActionSelector = action_sel_class(self.agent_par['action_selection'])
        self.noisy = self.agent_par['action_selection']['noisy']
        if self.noisy:
            assert isinstance(self.action_selector, GreedyAS)    #if noisy is true the selector SHOULD be greedy
        
        # Instatiate deep network model 
        if self.checkpoint is not None :
            self.qnetwort = self.load(self.checkpoint)     
        else:
            model_class = getattr(model_classes,self.agent_par['model_class'])
            self.qnetwork = model_class(self.obs_shape,self.action_size,self.lr,self.noisy).get_compiled_model()
            
        #if double qNetwork instantiate target-model also 
        if self.agent_par['double']:
            model_class = getattr(model_classes,self.agent_par['model_class'])
            self.qnetwork_target = model_class(self.obs_shape,self.action_size,self.lr,self.noisy).get_model()
            self.qnetwork_target.set_weights(self.qnetwork.get_weights())  
         

    # TODO : UNCOMMENT THIS AND COMMENT THE ACT METHOD BELOW TO USE ACTION SELECTORS (SINCE IT ALWAYS MAKE THE MODEL PREDICT IT IS CONSIDERABLY SLOWER) 
    # def act(self, obs, eval_mode) -> int : 
    #     state = tf.expand_dims(obs, axis=0)
    #     values = self.qnetwork.predict(state)
    #     action, is_best = self.action_selector.select_action(values, eval_mode)

    #     stats.utils_stats['exploration'] += 1-int(is_best)    #LOG

    #     return action 
    
    def act(self, obs, eval_mode : bool) -> int : 
        state = tf.expand_dims(obs, axis=0)

        if np.random.random() < self.action_selector.get_current_par_value() and not eval_mode: 
            stats.utils_stats['exploration'] += 1    #LOG
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.qnetwork.predict(state))
    

    def step(self, obs, action, reward, next_obs, done, agent):  
        self.t_step +=1

        #store one step experience in replay buffer
        self.memory.store_experience(obs,action,reward,next_obs,done)

        # Learn every UPDATE_EVERY time steps if enough samples are available in memory
        if self.t_step  % self.update_every == 0 and len(self.memory) >= self.sample_size:
            self.learn()

    def learn(self):
        state_sample, action_sample, reward_sample, next_state_sample, done_sample, is_weights = self.memory.sample_memory(self.sample_size)

        sample_indexes = np.arange(self.sample_size)

        q_targets = self.qnetwork.predict(state_sample)
        target_old = np.array(q_targets)

        if self.agent_par['double']:
            q_next_values = self.qnetwork_target.predict(next_state_sample)[sample_indexes, np.argmax(self.qnetwork.predict(next_state_sample), axis=1)]
        else:
            q_next_values = np.max(self.qnetwork.predict(next_state_sample), axis=1)

        q_targets[sample_indexes,action_sample] = reward_sample + ((1 - done_sample) * self.gamma * q_next_values)
        target_new = np.array(q_targets)
        
        if self.mem_is_PER:
            abs_errors = np.abs(target_old[sample_indexes,action_sample]-target_new[sample_indexes,action_sample])
            self.memory.buffer_update(abs_errors)

        history = self.qnetwork.fit(state_sample, q_targets, sample_weight = is_weights, batch_size = 32, verbose = 0)  
        stats.utils_stats['ep_losses'].append(history.history['loss'][0])

        if self.agent_par['double']:
            self.target_update(self.agent_par['tau'])
    
    def target_update(self, tau = 0.5e-3):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for t, e in zip(self.qnetwork_target.trainable_variables, self.qnetwork.trainable_variables):
            t.assign(t * (1.0 - tau) + e * tau)

    
    def on_episode_start(self):
        stats.log_stats['decaying_par'] = self.action_selector.get_current_par_value()    
        stats.utils_stats['ep_losses'] = []
        stats.utils_stats['exploration'] = 0

    def on_episode_end(self, agents):
        self.action_selector.decay()  
            
    def load(self,filename):
        print('loading model from checkpoints/'+filename)
        return keras.models.load_model('checkpoints/'+filename)

    def save(self,filename):
        print('saving model to checkpoints/'+filename)
        self.qnetwork.save('checkpoints/'+filename)  
    
    def __str__(self):
        return 'DQNAgent'


class PPOAgent(Agent):

    def load_params(self): 
        
        if not self.eval_mode :
            self.lr = self.agent_par['learning_rate']                 
            
            #Instantiate bufferReplay object 
            self.memory : PPOAgentBuffer = PPOAgentBuffer()   

        self.eps_clip = self.agent_par['surrogate_eps_clip'] 
        self.loss_weight = self.agent_par['loss_weight'] 
        self.entropy_weight= self.agent_par['entropy_weight'] 
        self.entropy_decay = self.agent_par['entropy_decay'] 

        self._optimizer = Adam(learning_rate=self.lr)

        # Instatiate ppo model 
        if self.checkpoint is not None :
            self.pponetwork = self.load(self.checkpoint)     
        else:
            self.pponetwork = model_classes.PPOModel(self.obs_shape,self.action_size)
         
    def act(self, obs, eval_mode : bool ):
        action, value = self.pponetwork.action_value(obs.reshape(1, -1))
        self._last_value = value
        return action.numpy()[0]

    def step(self, obs, action, reward, next_obs, done, agent):

        _, policy_logits = self.pponetwork(obs.reshape(1, -1))

        if self._last_value is None:
            _, self._last_value = self.pponetwork.action_value(obs.reshape(1, -1))

        self.memory.store_agent_experience(agent, action, self._last_value[0], obs, reward, done, policy_logits) 
        self._last_value = None

    def learn(self, agents):
        
        for agent in agents:
            actions, values, states, rewards, dones, probs = self.memory.get_agent_experience(agent) 
            

            _, next_value = self.pponetwork.action_value(states[-1].reshape(1, -1))
            discounted_rewards, advantages = self.get_advantages(rewards, dones, values, next_value[0])

            actions = tf.squeeze(tf.stack(actions))
            probs = tf.nn.softmax(tf.squeeze(tf.stack(probs)))
            action_inds = tf.stack([tf.range(0, actions.shape[0]), tf.cast(actions, tf.int32)], axis=1)
            
            old_probs = tf.gather_nd(probs, action_inds),

            with tf.GradientTape() as tape:
                values, policy_logits = self.pponetwork(tf.stack(states))
                act_loss = self.actor_loss(advantages, old_probs, action_inds, policy_logits)
                ent_loss = self.entropy_loss(policy_logits, self.entropy_weight)
                c_loss = self.critic_loss(discounted_rewards, values)
                tot_loss = act_loss + ent_loss + c_loss
                
                #log loss
                stats.utils_stats['ep_losses'].append(tot_loss)

            # Backpropagation
            grads = tape.gradient(tot_loss, self.pponetwork.trainable_variables)
            self._optimizer.apply_gradients(zip(grads, self.pponetwork.trainable_variables))
           
        self.entropy_weight = self.entropy_decay * self.entropy_weight
        self.memory.reset_mem() 

    def get_advantages(self, rewards, dones, values, next_value):
        discounted_rewards = np.array(rewards.tolist() + [next_value[0]])

        for t in reversed(range(len(rewards))):
            discounted_rewards[t] = rewards[t] + 0.99 * discounted_rewards[t+1] * (1-dones[t])
        discounted_rewards = discounted_rewards[:-1]
        # advantages are bootstrapped discounted rewards - values, using Bellman's equation
        advantages = discounted_rewards - values 
        # standardise advantages
        advantages -= np.mean(advantages)
        advantages /= (np.std(advantages) + 1e-10)
        # standardise rewards too
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)
        return discounted_rewards, advantages

    def actor_loss(self, advantages, old_probs, action_inds, policy_logits):
        probs = tf.nn.softmax(policy_logits)
        new_probs = tf.gather_nd(probs, action_inds)

        ratio = new_probs / old_probs

        policy_loss = -tf.reduce_mean(tf.math.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
         ))
        return policy_loss

    def critic_loss(self, discounted_rewards, value_est):
        return tf.cast(tf.reduce_mean(keras.losses.mean_squared_error(discounted_rewards, value_est)) * self.loss_weight,tf.float32)


    def entropy_loss(self, policy_logits, ent_discount_val):
        probs = tf.nn.softmax(policy_logits)
        entropy_loss = -tf.reduce_mean(keras.losses.categorical_crossentropy(probs, probs))
        return entropy_loss * ent_discount_val

    def save(self,filename):
        print('saving model to checkpoints/'+filename)
        self.pponetwork.save('checkpoints/'+filename) 
    
    def load(self,filename):
        print('loading model from checkpoints/'+filename)
        return keras.models.load_model('checkpoints/'+filename)


    def on_episode_start(self):
        stats.utils_stats['ep_losses'] = []

    def on_episode_end(self, agents):
        self.learn(agents)


    def __str__(self):
        return "ppo"