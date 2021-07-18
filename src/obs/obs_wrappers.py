from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from src.obs.obs_utils import split_tree_into_feature_groups, norm_obs_clip
from src.obs.new_obs import DensityForRailEnv

from flatland.envs.observations import TreeObsForRailEnv
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv


class Observation(ABC):
    def __init__(self, parameters):
        self.parameters = parameters
    
    @property
    def builder(self) -> ObservationBuilder:
        return self._builder
        
    @abstractmethod
    def get_obs_shape(self) -> Tuple: pass

    @abstractmethod
    def normalize(self, observation): pass


# Wrapper for the TreeObs observation 
class TreeObs(Observation):

    def __init__(self, parameters):
        super().__init__(parameters)
        predictor = None 
        if self.parameters['predictor'] :
            predictor = ShortestPathPredictorForRailEnv()
        self._builder = TreeObsForRailEnv(max_depth=self.parameters['tree_depth'],predictor=predictor)

    def get_obs_shape(self):
        # Compute the state size given the depth of the tree observation and the number of features
        n_features_per_node = self._builder.observation_dim
        n_nodes = 0
        for i in range(self.parameters['tree_depth'] + 1):
            n_nodes += np.power(4, i)
        return (n_features_per_node * n_nodes,)

    def normalize(self, observation):

        # This function normalizes the observation used by the RL algorithm
        data, distance, agent_data = split_tree_into_feature_groups(observation, self.parameters['tree_depth'])

        data = norm_obs_clip(data, fixed_radius=self.parameters['radius'])
        distance = norm_obs_clip(distance, normalize_to_range=True)
        agent_data = np.clip(agent_data, -1, 1)
        normalized_obs = np.concatenate((np.concatenate((data, distance)), agent_data))
        return normalized_obs


# Wrapper for the DensityObs observation 
class DensityObs(Observation):

    def __init__(self, parameters):
        super().__init__(parameters)

        self._h = self.parameters['height']
        self._w = self.parameters['width']
       
        self._builder = DensityForRailEnv(height=self._h,width=self._w)

    def get_obs_shape(self):
        # Compute the state size given the depth of the tree observation and the number of features
        return (self._h,self._w,2)           #2 is depth      

    def normalize(self, observation):
        
        # Change the observation shape in order to be stored in the replay buffer 
        density_agent, density_others = observation[0],observation[1]             #get the two element in the list 
        
        flat_d_a, flat_d_o = density_agent.flatten(), density_others.flatten()    #flatten each matrix to be stored in buffer replay

        normalized_obs = np.concatenate((flat_d_a,flat_d_o))                      #concatenate two arrays (they will be reshaped after)
 
        return normalized_obs