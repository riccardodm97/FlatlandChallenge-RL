from typing import Optional, List, Dict

import gym
import numpy as np

from flatland.core.env import Environment
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv


class DensityForRailEnv(ObservationBuilder):

    def __init__(self, height, width, max_t=10):
        super().__init__()
        self._height = height
        self._width = width
        self._depth = 1
        
        self._encode = lambda t: np.exp(-t / np.sqrt(max_t))
        
        self._predictor = ShortestPathPredictorForRailEnv(max_t)


    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        get density maps for agents and compose the observation with agent's and other's density maps
        """
        self._predictions = self._predictor.get()
        density_maps = dict()
        for handle in handles:
            density_maps[handle] = self.get(handle)
        obs = dict()
        for handle in handles:
            other_dens_maps = [density_maps[key] for key in density_maps if key != handle]
            others_density = np.mean(np.array(other_dens_maps), axis=0)
            obs[handle] = [density_maps[handle], others_density]
        return obs

    def get(self, handle: int = 0):
        """
        compute density map for agent: a value is assigned to every cell along the shortest path between
        the agent and its target based on the distance to the agent, i.e. the number of time steps the
        agent needs to reach the cell, encoding the time information.
        """
        density_map = np.zeros(shape=(self._height, self._width, self._depth), dtype=np.float32)
        if self._predictions[handle] is not None:
            for t, prediction in enumerate(self._predictions[handle]):
                if np.isnan(prediction).any():                                       #if any prediction from the predictor contains nan don't compute the density map and leave all zeros 
                    break
                p = tuple(np.array(prediction[1:3]).astype(int))
                d = t if self._depth > 1 else 0
                density_map[p][d] = self._encode(t)
        return density_map

    def set_env(self, env: Environment):
        self.env: RailEnv = env
        self._predictor.set_env(self.env)

    def reset(self):
        pass