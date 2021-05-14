import random

import wandb
from utils.timer import Timer

import matplotlib.pyplot as plt
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

import src.obs_wrappers as obs_wrap_classes
import src.agents as agent_classes
from src.obs_wrappers import Observation
from src.agents import Agent

import src.handlers.stats_handler as stats

class ExcHandler:
    def __init__(self, params, training=True, checkpoint=None):
        self._env_params = params['env'] # Environment
        self._obs_params = params['obs'] # Observation
        self._agn_params = params['agn'] # Agent
        self._trn_params = params['trn'] # Training

        self._training = training

        # Instantiate observation and environment 
        obs_wrap_class = getattr(obs_wrap_classes, self._obs_params['class'])
        self.obs_wrapper : Observation  = obs_wrap_class(self._obs_params) 
        self.env = self.initEnv(self.obs_wrapper.builder)

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._obs_size = self.obs_wrapper.get_obs_dim()

        # Instantiate agent 
        agent_class = getattr(agent_classes, self._agn_params['class'])
        self.agent : Agent = agent_class(self._obs_size, self._action_size, self._agn_params, self._trn_params, checkpoint , not self._training)
        
        # Max number of steps per episode as defined by flatland 
        self._max_steps = int(4 * 2 * (self._env_params['x_dim'] + self._env_params['y_dim'] + (self._env_params['n_agents'] / self._env_params['n_cities'])))

        #LOG
        wandb.config.max_steps = self._max_steps
        wandb.config.action_size = self._action_size
        wandb.config.obs_size = self._obs_size


    def initEnv(self, obs_builder):
        
        #setup the environment
        env = RailEnv(
            width=self._env_params['x_dim'],
            height=self._env_params['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=self._env_params['n_cities'],
                seed=self._env_params['seed'],
                grid_mode=True,
                max_rails_between_cities=self._env_params['max_rails_between_cities'],
                max_rails_in_city=self._env_params['max_rails_in_city']
            ),
            schedule_generator=sparse_schedule_generator(),
            number_of_agents=self._env_params['n_agents'],
            obs_builder_object=obs_builder
        )

        return env 
    
    def start(self, n_episodes):

        random.seed(self._env_params['seed'])
        np.random.seed(self._env_params['seed'])

        if self._training :
            self.train_agent(n_episodes)
        else:
            self.eval_agent(n_episodes)
        
    
    def train_agent(self,n_episodes):

        self.env.reset(True,True)

        agent_obs = [None] * self.env.get_num_agents()
        agent_prev_obs = [None] * self.env.get_num_agents()
        agent_prev_action = [2] * self.env.get_num_agents()       #TODO perché??
        update_values = [False] * self.env.get_num_agents()
        action_dict = dict()
        
        #LOG
        training_timer = Timer()                                  
        training_timer.start()
        
        for ep_id in range(n_episodes):
            
            #LOG
            stats.action_count = [0] * self._action_size
            stats.ep_score = 0.0
            stats.min_steps_to_complete = self._max_steps

            # Reset environment
            obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

            for handle in self.env.get_agent_handles():
                if obs[handle]:
                    agent_obs[handle] = self.obs_wrapper.normalize(obs[handle])
                    agent_prev_obs[handle] = agent_obs[handle].copy()


            for step in range(self._max_steps-1):
                for handle in self.env.get_agent_handles():
                    if info['action_required'][handle]:
                        update_values[handle] = True
                        action = self.agent.act(agent_obs[handle])
                        stats.action_count[action] +=1
                    else :
                        update_values[handle] = False
                        action = 0        
                    action_dict.update({handle: action})
                    
                # Environment step
                next_obs, all_rewards, done, info = self.env.step(action_dict)

                # Update replay buffer and train agent
                for handle in self.env.get_agent_handles():
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if update_values or done['__all__']:
                        # Add state to memory
                        self.agent.step(
                            obs = agent_prev_obs[handle],
                            action = agent_prev_action[handle],
                            reward = all_rewards[handle],
                            next_obs = agent_obs[handle],
                            done = done[handle]
                        )
                        
                        agent_prev_obs[handle] = agent_obs[handle].copy()
                        agent_prev_action[handle] = action_dict[handle]

                    if next_obs[handle]:
                        agent_obs[handle] = self.obs_wrapper.normalize(next_obs[handle])

                    stats.ep_score += all_rewards[handle]
                
                if True in done.values() and stats.min_steps_to_complete==self._max_steps:
                    stats.min_steps_to_complete = step +1

                if done['__all__']:
                    break
            
            self.agent.on_episode_end()

            # Collection information about training after each episode
            stats.episode_stats['completion_perc'] = np.sum([int(done[idx]) for idx in self.env.get_agent_handles()]) / max(1, self.env.get_num_agents())
            stats.episode_stats['norm_score'] = stats.ep_score / (self._max_steps * self.env.get_num_agents())
            #stats.episode_stats['action_probs'] = np.histogram(np.divide(stats.action_count,np.sum(stats.action_count))) 
            stats.episode_stats['min_step_to_complete'] = stats.min_steps_to_complete

            print(
                '\rTraining {} agents \t Episode {}\t Average Score: {:.3f}\t Dones: {:.2f}%'.format(
                    self.env.get_num_agents(),
                    ep_id,
                    stats.episode_stats['norm_score'],
                    stats.episode_stats['completion_perc']*100
                ))
            stats.on_episode_end(ep_id)
        
        self.agent.save('checkpoints/' + wandb.run.id)
        

    def eval_agent(self,n_episodes):

        self.env.reset(True,True)

        env_renderer = RenderTool(self.env)

        action_dict = dict()

        for _ in range(n_episodes):
            agent_obs = [None] * self.env.get_num_agents()

            # Reset environment
            obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

            env_renderer.reset()

            for _ in range(self._max_steps-1):
                for handle in self.env.get_agent_handles():
                    if obs[handle] :
                        agent_obs[handle] = self.obs_wrapper.normalize(obs[handle])

                    action = 0    
                    if info['action_required'][handle]:
                        action = self.agent.act(agent_obs[handle])

                    action_dict.update({handle: action})
                    
                # Environment step
                obs, all_rewards, done, info = self.env.step(action_dict)

                env_renderer.render_env(show=True, show_observations=True, show_predictions=False)


                if done['__all__']:
                    break
        
        env_renderer.close_window()
        