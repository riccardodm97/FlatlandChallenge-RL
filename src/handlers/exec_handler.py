import random
from typing import Tuple

import wandb
from utils.timer import Timer

import matplotlib.pyplot as plt
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

import src.obs_wrappers as obs_wrap_classes
import src.agents as agent_classes
from src.obs_wrappers import Observation
from src.agents import Agent

import src.handlers.stats_handler as stats

class ExcHandler:
    def __init__(self, agn_par : dict, env_par : dict, mode : str, checkpoint : str = None):
        self._agn_par = agn_par # Agent
        self._env_par = env_par # Environment

        self._mode = mode
        self._checkpoint = checkpoint

        # Instantiate observation and environment 
        self._obs_wrapper, self._env, self._max_steps = self.handleEnv(self._env_par)

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._obs_size = self._obs_wrapper.get_obs_dim()

        # Instantiate agent 
        self._agent = self.handleAgent(self._agn_par)
        
        #LOG
        wandb.config.max_steps = self._max_steps
        wandb.config.action_size = self._action_size
        wandb.config.obs_size = self._obs_size


    def handleEnv(self, environment_param : dict) -> Tuple[Observation,RailEnv,int]:

        env_par : dict = environment_param['env']
        obs_par : dict = environment_param['obs']

        # Instantiate observation 
        obs_wrap_class = getattr(obs_wrap_classes, obs_par['class'])
        obs_wrapper : Observation  = obs_wrap_class(obs_par) 

        #init malfunction parameters
        malfunction = None
        if env_par['malfunction']['enabled']:
            malfunction = malfunction_from_params(MalfunctionParameters(malfunction_rate=env_par['malfunction']['rate'],
                                                                        min_duration=env_par['malfunction']['min_step'],
                                                                        max_duration=env_par['malfunction']['max_step'])) 
        #init speed map 
        speed_map = None
        if env_par['diff_speed_enabled']:
            speed_map = {1.  : 0.25,  # Fast passenger train
                        1./2.: 0.25,  # Fast freight train
                        1./3.: 0.25,  # Slow commuter train
                        1./4.: 0.25}  # Slow freight train

        #TODO add prediction_builder
        
        #setup the environment
        env = RailEnv(
            width=env_par['x_dim'],
            height=env_par['y_dim'],
            rail_generator=sparse_rail_generator(
                max_num_cities=env_par['n_cities'],
                seed=env_par['seed'],
                grid_mode=env_par['grid'],
                max_rails_between_cities=env_par['max_rails_between_cities'],
                max_rails_in_city=env_par['max_rails_in_city']
            ),
            schedule_generator=sparse_schedule_generator(speed_ratio_map=speed_map),
            number_of_agents=env_par['n_agents'],
            obs_builder_object= obs_wrapper.builder,
            random_seed=env_par['seed'],
            malfunction_generator_and_process_data = malfunction
        )

        # Max number of steps per episode as defined by flatland 
        max_steps = int(4 * 2 * (env_par['x_dim'] + env_par['y_dim'] + (env_par['n_agents'] / env_par['n_cities'])))

        return obs_wrapper, env, max_steps
    
    def handleAgent(self, agn_par : dict) -> Agent:
        agent_class = getattr(agent_classes, agn_par['class'])
        agent : Agent = agent_class(self._obs_size, self._action_size, agn_par, self._checkpoint, True if self._mode == 'eval' else False)

        return agent
    
    def start(self, n_episodes, show = False):

        random.seed(self._env_par['env']['seed'])
        np.random.seed(self._env_par['env']['seed'])

        if self._mode == 'train' :
            self.train_agent(n_episodes)
        elif self._mode == 'eval':
            self.eval_agent(n_episodes, show)
        else : raise ValueError('ERROR: mode should be either train or eval')
        
    
    def train_agent(self,n_episodes):

        self._env.reset(True,True)

        agent_obs = [None] * self._env.get_num_agents()
        agent_prev_obs = [None] * self._env.get_num_agents()
        agent_prev_action = [2] * self._env.get_num_agents()       #TODO perché??
        update_values = [False] * self._env.get_num_agents()
        action_dict = dict()

        smoothing = 0.99
        stats.utils_stats['smoothed_ep_score'] = -1
        stats.utils_stats['smoothed_dones'] = 0.0
        
        for ep_id in range(n_episodes):

            # Initialize agent 
            self._agent.on_episode_start()
            
            # LOG
            stats.utils_stats['action_count'] = [0] * self._action_size
            stats.utils_stats['ep_score'] = 0.0
            stats.utils_stats['min_steps_to_complete'] = self._max_steps

            # Reset environment
            obs, info = self._env.reset(regenerate_rail=True, regenerate_schedule=True)

            for handle in self._env.get_agent_handles():
                if obs[handle]:
                    agent_obs[handle] = self._obs_wrapper.normalize(obs[handle])
                    agent_prev_obs[handle] = agent_obs[handle].copy()


            for step in range(self._max_steps-1):
                for handle in self._env.get_agent_handles():
                    if info['action_required'][handle]:
                        update_values[handle] = True
                        action = self._agent.act(agent_obs[handle])
                        stats.utils_stats['action_count'][action] +=1
                    else :
                        update_values[handle] = False
                        action = 0        
                    action_dict.update({handle: action})
                    
                # Environment step
                next_obs, all_rewards, done, info = self._env.step(action_dict)

                # Update replay buffer and train agent
                for handle in self._env.get_agent_handles():
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if update_values or done['__all__']:
                        # Add state to memory
                        self._agent.step(
                            obs = agent_prev_obs[handle],
                            action = agent_prev_action[handle],
                            reward = all_rewards[handle],
                            next_obs = agent_obs[handle],
                            done = done[handle]
                        )
                        
                        agent_prev_obs[handle] = agent_obs[handle].copy()
                        agent_prev_action[handle] = action_dict[handle]

                    if next_obs[handle]:
                        agent_obs[handle] = self._obs_wrapper.normalize(next_obs[handle])

                    stats.utils_stats['ep_score'] += all_rewards[handle]
                
                stats.utils_stats['min_steps_to_complete'] = step

                if done['__all__']:
                    break

            self._agent.on_episode_end()
            
            # LOG 
            stats.completion_window.append(np.sum([int(done[idx]) for idx in self._env.get_agent_handles()]) / max(1, self._env.get_num_agents()))
            stats.score_window.append(stats.utils_stats['ep_score'] / (self._max_steps * self._env.get_num_agents()))
            stats.min_steps_window.append(stats.utils_stats['min_steps_to_complete'])

            stats.utils_stats['smoothed_ep_score'] = stats.utils_stats['smoothed_ep_score'] * smoothing + stats.utils_stats['ep_score'] * (1.0 - smoothing)
            stats.utils_stats['dones'] = np.sum([int(done[idx]) for idx in self._env.get_agent_handles()]) / max(1, self._env.get_num_agents())
            stats.utils_stats['smoothed_dones'] = stats.utils_stats['smoothed_dones'] * smoothing + stats.utils_stats['dones'] * (1.0 - smoothing)

            stats.log_stats['score'] = stats.utils_stats['ep_score']
            stats.log_stats['smoothed_score'] = stats.utils_stats['smoothed_ep_score']
            stats.log_stats['average_score'] = np.mean(stats.score_window)
            stats.log_stats['dones'] = stats.utils_stats['dones']
            stats.log_stats['smoothed_dones'] = stats.utils_stats['smoothed_dones']
            stats.log_stats['average_dones'] = np.mean(stats.completion_window)
            stats.log_stats['min_step_to_complete'] = stats.utils_stats['min_steps_to_complete']
            stats.log_stats['average_min_step_to_complete'] = np.mean(stats.min_steps_window)

            print(
                '\r🚂 Training {} agents' 
                '\t 🏁 Episode {}'
                '\t 🏆 Score: {}'
                ' Smoothed: {:.3f}'
                ' Average: {:.3f}'
                '\t 💯 Dones: {:.2f}'
                ' Smoothed: {:.2f}'
                ' Average: {:.2f}%'
                '\t 🧭 N° steps: {}'
                ' Average: {:.2f}'
                '\t 🎲 Epsilon: {:.3f}'.format(
                    self._env.get_num_agents(),
                    ep_id,
                    stats.log_stats['score'],
                    stats.log_stats['smoothed_score'],
                    stats.log_stats['average_score'],
                    stats.log_stats['dones'],
                    stats.log_stats['smoothed_dones'],
                    stats.log_stats['average_dones']*100,
                    stats.log_stats['min_step_to_complete'],
                    stats.log_stats['average_min_step_to_complete'],
                    stats.log_stats['eps']
                ))

            stats.on_episode_end(ep_id)

        self.checkpoint()

    def checkpoint(self):
          self._agent.save(wandb.run.id)

    def eval_agent(self, n_episodes, show):

        self._env.reset(True,True)

        if show : env_renderer = RenderTool(self._env)

        action_dict = dict()

        for _ in range(n_episodes):
            agent_obs = [None] * self._env.get_num_agents()

            # Reset environment
            obs, info = self._env.reset(regenerate_rail=True, regenerate_schedule=True)

            if show : env_renderer.reset()

            for _ in range(self._max_steps-1):
                for handle in self._env.get_agent_handles():
                    if obs[handle] :
                        agent_obs[handle] = self._obs_wrapper.normalize(obs[handle])

                    action = 0    
                    if info['action_required'][handle]:
                        action = self._agent.act(agent_obs[handle])

                    action_dict.update({handle: action})
                    
                # Environment step
                obs, all_rewards, done, info = self._env.step(action_dict)

                if show : env_renderer.render_env(show=True, show_observations=True, show_predictions=False)


                if done['__all__']:
                    break
        
        if show : env_renderer.close_window()
        