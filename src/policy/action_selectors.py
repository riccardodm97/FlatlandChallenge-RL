from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class ActionSelector(ABC):

    def __init__(self, parameters : dict):

        self._parameter_start = parameters['p_start']
        self._parameter_end = parameters['p_end']        
        self._parameter_decay = parameters['p_decay']
    
    @abstractmethod
    def select_action(self, action_values, eval_mode) -> Tuple[int,bool]: pass

    @abstractmethod
    def decay(self): pass

    @abstractmethod
    def get_current_par_value(self) -> float: pass

class RandomAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def select_action(self, action_values, eval_mode) -> Tuple[int,bool]:
        return np.random.choice(action_values.size), False
    
    def decay(self):
        pass
    
    def get_current_par_value(self):
        pass

class GreedyAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def select_action(self, action_values, eval_mode) -> Tuple[int,bool]:
        raise NotImplementedError  
    
    def decay(self):
        pass         
    
    def get_current_par_value(self):
        return 0.0                       


class EpsilonGreedyAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
        self._epsilon = self._parameter_start 
    
    def select_action(self, action_values, eval_mode : bool) -> Tuple[int,bool]:
        max_action = np.argmax(action_values)    
        if  not eval_mode and np.random.random() < self._epsilon :
            rnd_action = np.random.choice(action_values.size)
            return rnd_action, rnd_action==max_action
        else:
           return max_action, True
    
    def decay(self):
        self._epsilon = max(self._parameter_end, self._parameter_decay*self._epsilon)    
    
    def get_current_par_value(self):
        return self._epsilon


class BoltzmannAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
        self._temperature = self._parameter_start
    
    def select_action(self, action_values, eval_mode) -> Tuple[int,bool]:
        max_action = np.argmax(action_values)
        if eval_mode:
            return max_action, True
        
        val = action_values.copy()
        exps = np.exp(val / self._temperature)
        boltz_prob = exps / np.sum(exps, axis=1)
        boltz_prob = boltz_prob[0]                #need to be 1d array 

        rnd_action = np.random.choice(action_values.size,p=boltz_prob)
        return rnd_action, rnd_action==max_action 

    def decay(self):
        self._temperature = max(self._parameter_end, self._parameter_decay*self._temperature) 
    
    def get_current_par_value(self):
        return self._temperature

    






