import random
import matplotlib.pyplot as plt
import numpy as np

from collections import deque
from pathlib import Path

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool

import src.obs_wrappers as obs_wrap_classes
import src.agents as agent_classes



class ExcHandler:
    def __init__(self, params, training=True, checkpoint=None):
        self._env_params = params['env'] # Environment
        self._trn_params = params['trn'] # Training
        self._obs_params = params['obs'] # Observation
        self._agn_params = params['agn'] # Agent

        self._training = training

        # Instantiate observation and environment 
        obs_wrap_class = getattr(obs_wrap_classes, self._obs_params['class'])
        self.obs_wrapper = obs_wrap_class(self._obs_params)
        self.env = self.initEnv(self.obs_wrapper.builder)

        # The action space of flatland is 5 discrete actions
        self._action_size = 5
        self._obs_size = self.obs_wrapper.get_obs_dim()

        # Instantiate agent 
        agent_class = getattr(agent_classes, self._agn_params['class'])
        self.agent = agent_class(self._obs_size, self._action_size, self._agn_params, self._trn_params, checkpoint)
        
        # Max number of steps per episode as defined by flatland 
        self._max_steps = int(4 * 2 * (self._env_params['x_dim'] + self._env_params['y_dim'] + (
                self._env_params['n_agents'] / self._env_params['n_cities'])))


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
            self.run_episodes(n_episodes)
        else:
            pass #TODO : implementare ciclo di sola visualizzazione (usare render)
        
    
    def run_episodes(self,n_episodes):

        self.env.reset(True,True)


        scores_window = deque(maxlen=100)  
        completion_window = deque(maxlen=100)
        scores = []
        completion = []
        action_count = [0] * self.action_size

        agent_obs = [None] * self.env.get_num_agents()
        agent_prev_obs = [None] * self.env.get_num_agents()
        agent_prev_action = [2] * self.env.get_num_agents()
        update_values = [False] * self.env.get_num_agents()
        action_dict = dict()
        
        for n in range(n_episodes):

            # Initialize episode
            n_steps = 0
            score = 0

            # Reset environment
            obs, info = self.env.reset(regenerate_rail=True, regenerate_schedule=True)

            for handle in self.env.get_agent_handles():
                if obs[handle]:
                    agent_obs[handle] = self.obs_wrapper.normalize_observation(obs[handle])
                    agent_prev_obs[handle] = agent_obs[handle].copy()


            for step in range(self._max_steps-1):
                for handle in self.env.get_agent_handles():
                    if info['action_required'][handle]:
                        update_values[handle] = True
                        action = self.agent.act(agent_obs[handle])
                        action_count[action] +=1
                    else :
                        update_values[handle] = False
                        action = 0        #TODO modificare
                    action_dict.update({handle: action})
                    
                # Environment step
                next_obs, all_rewards, done, info = self.env.step(action_dict)

                # Update replay buffer and train agent
                for handle in self.env.get_agent_handles():
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if update_values or done['__all__']:
                        # Add state to memory
                        self.agent.step(
                            state = agent_prev_obs[handle],
                            action = agent_prev_action[handle],
                            reward = all_rewards[handle],
                            next_state = agent_obs[handle],
                            done = done[handle])

                        agent_prev_obs[handle] = agent_obs[handle].copy()
                        agent_prev_action[handle] = action_dict[handle]

                    if next_obs[handle]:
                        agent_obs[handle] = self.obs_wrapper.normalize_observation(next_obs[handle])

                    score += all_rewards[handle]

                if done['__all__']:
                    break

            # Collection information about training
            tasks_finished = np.sum([int(done[idx]) for idx in self.env.get_agent_handles()])
            completion_window.append(tasks_finished / max(1, self.env.get_num_agents()))
            scores_window.append(score / (self._max_steps * self.env.get_num_agents()))
            completion.append((np.mean(completion_window)))
            scores.append(np.mean(scores_window))
            action_probs = action_count / np.sum(action_count)

            print(
                '\rTraining {} agents \t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\t Action Probabilities: \t {}'.format(
                    self.env.get_num_agents(),
                    n,
                    np.mean(scores_window),
                    100 * np.mean(completion_window),
                    action_probs
                ))

        # Plot overall training progress at the end
        plt.plot(scores)
        plt.show()

        plt.plot(completion)
        plt.show()
