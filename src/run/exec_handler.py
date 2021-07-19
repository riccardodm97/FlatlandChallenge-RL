import random
from typing import Tuple

import wandb
from src.utils.timer import Timer

import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

import src.obs.obs_wrappers as obs_wrap_classes
import src.policy.agents as agent_classes
from src.obs.obs_wrappers import Observation
from src.policy.agents import Agent

import src.utils.stats_handler as stats

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
        self._obs_shape = self._obs_wrapper.get_obs_shape()

        # Instantiate agent 
        self._agent = self.handleAgent(self._agn_par)
        
        #LOG
        wandb.config.max_steps = self._max_steps
        wandb.config.action_size = self._action_size
        wandb.config.obs_shape = str(self._obs_shape)



    def handleEnv(self, environment_param : dict) -> Tuple[Observation,RailEnv,int]:

        env_par : dict = environment_param['env']
        obs_par : dict = environment_param['obs']

        #if densityObs, the observation shape should be the same as the env grid shape 
        if obs_par['class'] == 'DensityObs':
            assert env_par['x_dim'] == obs_par['width']
            assert env_par['y_dim'] == obs_par['height']

        #instantiate observation 
        obs_wrap_class = getattr(obs_wrap_classes, obs_par['class'])
        obs_wrapper : Observation  = obs_wrap_class(obs_par) 

        #the prediction_builder is added to the Observation

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

        if self._env_par['obs']['class'] == 'DensityObs' :                       #if observation is DensityObs the model SHOULD be DuelingCNN
            assert agn_par['model_class'] == 'DuelingCNN'

        agent_class = getattr(agent_classes, agn_par['class'])
        agent : Agent = agent_class(self._obs_shape, self._action_size, agn_par, self._checkpoint, True if self._mode == 'eval' else False)

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
        agent_prev_action = [2] * self._env.get_num_agents()       
        update_values = [False] * self._env.get_num_agents()
        action_dict = dict()

        eval_while_train = False                                   # do some eval pass or not while training  

        stats.utils_stats['smoothed_score'] = -1
        stats.utils_stats['smoothed_completion'] = 0.0

        smoothed_eval_normalized_score = -1.0
        smoothed_eval_completion = 0.0
        smoothing = 0.9

        learn_timer = Timer()
        act_timer = Timer()
        
        for ep_id in range(n_episodes):

            # Initialize agent 
            self._agent.on_episode_start()
            
            # LOG
            stats.utils_stats['action_count'] = 0
            stats.utils_stats['ep_score'] = 0.0
            stats.utils_stats['steps_first_to_complete'] = self._max_steps
            stats.utils_stats['steps_last_to_complete'] = 0
            stats.utils_stats['learn_time_steps'] = []
            stats.utils_stats['act_time_steps'] = []

            learn_timer.reset()
            act_timer.reset()

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
                        act_timer.start_reset()
                        action = self._agent.act(agent_obs[handle],False)
                        act_timer.end()
                        stats.utils_stats['act_time_steps'].append(act_timer.get())
                        stats.utils_stats['action_count']+=1
                    else :
                        update_values[handle] = False
                        action = 0        
                    action_dict.update({handle: action})
                    
                # Environment step
                next_obs, all_rewards, done, info = self._env.step(action_dict)

                # Update replay buffer and train agent
                for handle in self._env.get_agent_handles():
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if update_values[handle] or done['__all__']:
                        # Add state to memory
                        learn_timer.start_reset()
                        self._agent.step(
                            obs = agent_prev_obs[handle],
                            action = agent_prev_action[handle],
                            reward = all_rewards[handle],
                            next_obs = agent_obs[handle],
                            done = done[handle],
                            agent = handle
                        )
                        learn_timer.end()
                        stats.utils_stats['learn_time_steps'].append(learn_timer.get())
                        
                        agent_prev_obs[handle] = agent_obs[handle].copy()
                        agent_prev_action[handle] = action_dict[handle]

                    # Preprocess the new observations
                    if next_obs[handle]:
                        agent_obs[handle] = self._obs_wrapper.normalize(next_obs[handle])

                    stats.utils_stats['ep_score'] += all_rewards[handle]
                
                if True in done.values() and stats.utils_stats['steps_first_to_complete']==self._max_steps:
                    stats.utils_stats['steps_first_to_complete'] = step + 1

                stats.utils_stats['steps_last_to_complete'] = step + 1 

                if done['__all__']:
                    break

            self._agent.on_episode_end(self._env.get_agent_handles())

            # Evaluate policy and log results at some interval
            if eval_while_train and ((ep_id  % 100 == 0 and ep_id!=0) or ep_id == n_episodes-1) :
                scores, completions, nb_steps_eval = self.eval_agent(10,False)

                stats.log_stats["evaluation/scores_min"] = np.min(scores)
                stats.log_stats["evaluation/scores_max"] = np.max(scores)
                stats.log_stats["evaluation/scores_mean"] = np.mean(scores)
                stats.log_stats["evaluation/scores_std"] = np.std(scores)
                stats.log_stats["evaluation/completions_min"] = np.min(completions)
                stats.log_stats["evaluation/completions_max"] = np.max(completions)
                stats.log_stats["evaluation/completions_mean"] = np.mean(completions)
                stats.log_stats["evaluation/completions_std"] = np.std(completions)
                stats.log_stats["evaluation/nb_steps_min"] = np.min(nb_steps_eval)
                stats.log_stats["evaluation/nb_steps_max"] = np.max(nb_steps_eval)
                stats.log_stats["evaluation/nb_steps_mean"]=  np.mean(nb_steps_eval)
                stats.log_stats["evaluation/nb_steps_std"] = np.std(nb_steps_eval)

                smoothed_eval_normalized_score = smoothed_eval_normalized_score * smoothing + np.mean(scores) * (1.0 - smoothing)
                smoothed_eval_completion = smoothed_eval_completion * smoothing + np.mean(completions) * (1.0 - smoothing)
                stats.log_stats["evaluation/smoothed_score"] = smoothed_eval_normalized_score 
                stats.log_stats["evaluation/smoothed_completion"] = smoothed_eval_completion
            
            # LOGS
            stats.utils_stats['completion'] = np.sum([int(done[idx]) for idx in self._env.get_agent_handles()]) / max(1, self._env.get_num_agents())

            stats.steps_first_window.append(stats.utils_stats['steps_first_to_complete'] / self._max_steps)
            stats.steps_last_window.append(stats.utils_stats['steps_last_to_complete'] / self._max_steps)
            if stats.utils_stats.get('exploration') is not None : 
                stats.exploration_window.append(stats.utils_stats['exploration'] / max(1,stats.utils_stats['action_count']))
            stats.completion_window.append(stats.utils_stats['completion'])
            stats.score_window.append(stats.utils_stats['ep_score'] / (self._max_steps * max(1, self._env.get_num_agents())))
            stats.learn_timer_window.append(np.mean(stats.utils_stats['learn_time_steps']))
            stats.act_timer_window.append(np.mean(stats.utils_stats['act_time_steps']))

            stats.utils_stats['smoothed_score'] = stats.utils_stats['smoothed_score'] * smoothing + np.mean(stats.score_window) * (1.0 - smoothing)
            stats.utils_stats['smoothed_completion'] = stats.utils_stats['smoothed_completion'] * smoothing + np.mean(stats.completion_window) * (1.0 - smoothing)
        
            stats.log_stats['smoothed_score'] = stats.utils_stats['smoothed_score']
            stats.log_stats['average_score'] = np.mean(stats.score_window)
            stats.log_stats['smoothed_completion'] = stats.utils_stats['smoothed_completion']
            stats.log_stats['average_completion'] = np.mean(stats.completion_window)

            stats.log_stats['average_steps_first_to_complete'] = np.mean(stats.steps_first_window)
            stats.log_stats['average_steps_last_to_complete'] = np.mean(stats.steps_last_window)
            stats.log_stats['exploration'] = np.mean(stats.exploration_window)
            stats.log_stats['smoothed_learning_time'] = np.mean(stats.learn_timer_window)
            stats.log_stats['smoothed_acting_time'] = np.mean(stats.act_timer_window)
            stats.log_stats['mean_episode_loss'] = np.mean(stats.utils_stats['ep_losses'])
            stats.log_stats['std_episode_loss'] = np.std(stats.utils_stats['ep_losses'])
            stats.log_stats['buffer_level'] = len(self._agent.memory)
            
            print(
                '\r🚂 Training {} agents' 
                '\t 🏁 Episode {}'
                '\t 🏆 Score: {:.3f}'
                ' Avg: {:.3f}'
                '\t 💯 Completion: {:6.2f}%'
                ' Avg: {:6.2f}%'
                '\t 🧭 Avg N° steps: {:.2f}'.format(
                    self._env.get_num_agents(),
                    ep_id,
                    stats.utils_stats['ep_score'],
                    stats.log_stats['average_score'],
                    stats.utils_stats['completion']*100,
                    stats.log_stats['average_completion']*100,
                    np.mean([stats.utils_stats['steps_first_to_complete'] / self._max_steps, stats.utils_stats['steps_last_to_complete'] / self._max_steps])
                ))

            stats.on_episode_end(ep_id)

        self.checkpoint()

    def checkpoint(self):
          self._agent.save(wandb.run.id)

    def eval_agent(self, n_episodes, show):

        self._env.reset(True,True)

        if show : env_renderer = RenderTool(self._env)

        #stats
        scores = []
        completions = []
        nb_steps = []

        action_dict = dict()

        for _ in range(n_episodes):
            agent_obs = [None] * self._env.get_num_agents()

            score = 0.0

            # Reset environment
            obs, info = self._env.reset(regenerate_rail=True, regenerate_schedule=True)

            if show : env_renderer.reset()

            for step in range(self._max_steps-1):
                for handle in self._env.get_agent_handles():
                    if obs[handle] :
                        agent_obs[handle] = self._obs_wrapper.normalize(obs[handle])

                    action = 0    
                    if info['action_required'][handle]:
                        action = self._agent.act(agent_obs[handle],True)

                    action_dict.update({handle: action})
                    
                # Environment step
                obs, all_rewards, done, info = self._env.step(action_dict)

                for agent in self._env.get_agent_handles():
                    score += all_rewards[agent]
                
                final_step = step

                if show : env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

                if done['__all__']:
                    break
            
            normalized_score = score / (self._max_steps * self._env.get_num_agents())
            scores.append(normalized_score)

            tasks_finished = sum(done[idx] for idx in self._env.get_agent_handles())
            completion = tasks_finished / max(1, self._env.get_num_agents())
            completions.append(completion)

            nb_steps.append(final_step)
        
        if show : env_renderer.close_window()

        print(" ✅ Eval: score {:.3f} done {:.1f}%".format(np.mean(scores), np.mean(completions) * 100.0))

        return scores, completions, nb_steps
    


        