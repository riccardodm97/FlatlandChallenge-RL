from abc import ABC, abstractmethod

class ActionSelector(ABC):

    def __init__(self,parameters):

        self._parameter_start = parameters['p_start']
        self._parameter_end = parameters['p_end']        
        self._parameter_decay = parameters['p_decay']
    
    @abstractmethod
    def select_action(self,action_values): pass

    @abstractmethod
    def decay(self): pass

    @abstractmethod
    def get_current_par_value(self): pass

class GreedyAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def select_action(self, action_values):
        raise NotImplementedError
    
    def decay(self):
        raise NotImplementedError
    
    def get_current_par_value(self):
        raise NotImplementedError

class EpsilonGreedyAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
        self._epsilon = self._parameter_start
    
    def select_action(self, action_values):
        raise NotImplementedError
    
    def decay(self):
        raise NotImplementedError
    
    def get_current_par_value(self):
        raise NotImplementedError

class BoltzmannAS(ActionSelector):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def select_action(self, action_values):
        raise NotImplementedError
    
    def decay(self):
        raise NotImplementedError
    
    def get_current_par_value(self):
        raise NotImplementedError

    






