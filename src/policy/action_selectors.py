from abc import ABC, abstractmethod
import numpy as np

import src.utils.stats_handler as stats


class ActionSelector(ABC):

    def __init__(self, parameters : dict, eval_mode : bool = False):

        self._parameter_start = parameters['p_start']
        self._parameter_end = parameters['p_end']        
        self._parameter_decay = parameters['p_decay']

        self.eval_mode = eval_mode
    
    @abstractmethod
    def select_action(self, action_values) -> int: pass

    @abstractmethod
    def decay(self): pass

    @abstractmethod
    def get_current_par_value(self) -> float: pass

class RandomAS(ActionSelector):

    def __init__(self, parameters, eval_mode):
        super().__init__(parameters, eval_mode=eval_mode)
    
    def select_action(self, action_values) -> int:
        return np.random.choice(action_values.size)
    
    def decay(self):
        pass
    
    def get_current_par_value(self):
        pass

class GreedyAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def select_action(self, action_values):
        return np.argmax(action_values)  
    
    def decay(self):
        pass          # TODO check 
    
    def get_current_par_value(self):
        return 0      # TODO check 

class EpsilonGreedyAS(ActionSelector):

    def __init__(self, parameters, eval_mode):
        super().__init__(parameters,eval_mode)
        self._epsilon = self._parameter_start if self.eval_mode is False else 0.05
    
    def select_action(self, action_values) -> int:
        if np.random.random() < self._epsilon :
            stats.log_stats['random_action_taken'] += 1     #LOG 
            return np.random.choice(action_values.size)
        else:
            return np.argmax(action_values)     
    
    def decay(self):
        self._epsilon = max(self._parameter_end, self._parameter_decay*self._epsilon)    
    
    def get_current_par_value(self):
        return self._epsilon

class BoltzmannAS(ActionSelector):

    def __init__(self, parameters, eval_mode):
        super().__init__(parameters,eval_mode)
        self._temperature = self._parameter_start
    
    def select_action(self, action_values):
        if self.eval_mode is True :
            return np.argmax(action_values)
        
        val = action_values.copy()
        exps = np.exp(val / self._temperature)
        boltz_prob = exps / np.sum(exps, axis=1)
        boltz_prob = boltz_prob[0]                #should be 1d array 

        return np.random.choice(action_values.size,p=boltz_prob)

    def decay(self):
        self._temperature = max(self._parameter_end, self._parameter_decay*self._temperature) 
    
    def get_current_par_value(self):
        return self._temperature

    






