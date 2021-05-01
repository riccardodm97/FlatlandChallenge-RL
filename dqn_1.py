import numpy as np

import tensorflow as tf

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.observations import TreeObsForRailEnv


class Memory:

    def __init__(self,mem_size, input_shape):
        self.mem_size = mem_size
        self.stored = 0
        self.state_memory = np.zeros((self.mem_size,input_shape))
        self.action_memory = np.zeros((self.mem_size))
        self.reward_memory = np.zeors((self.mem_size))
        self.new_state_memory = np.zeros((self.mem_size,input_shape))
        self.not_done_memory = np.zeros((self.mem_size))
    
    def store_experience(self,state,action,reward,new_state,done):
        index = self.stored % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.not_done_memory[index] = 1 - int(done)
        self.stored += 1

    def sample_memory(self,sample_size):
        sample_indices = np.random.choice(np.min(self.mem_size,self.stored),size=sample_size)
        state_sample = self.state_memory[sample_indices]
        action_sample = self.action_memory[sample_indices]
        reward_sample = self.reward_memory[sample_indices]
        new_state_sample = self.new_state_memory[sample_indices]
        not_done_sample = self.not_done_memory[sample_indices] 

        return state_sample,action_sample,reward_sample,new_state_sample,not_done_sample

class DQN:

    def __init__(self,input_shape,ouput_shape,lr):
        self.input_shape = input_shape
        self.ouput_shape = ouput_shape
        self.lr = lr
    
    def build_model(self):

        #create model with tf
        self.model = [] #model
        pass
    
    def load_model(self,path):
        #load model with tf
        if path is None:
            pass
        pass
    
    def save_model(self,path):
        #save model to some path 
        pass

class Agent:
    
    def __init__(self, lr, gamma, n_actions, epsilon, epsilon_min,sample_size,input_shape, epsilon_decay=0.995, mem_size=1000000,model_path = None):

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.sample_size = sample_size
        self.memory = Memory(mem_size, input_shape)
        self.model = DQN(input_shape, n_actions,lr).load_model(model_path)

    
    def remember(self,state,action,reward,next_state,done):
        self.memory.store_experience(state,action,reward,next_state,done)
        self.epsilon = np.max(self.epsilon_min, self.epsilon_decay*self.epsilon) 
    
    def choose_action(self, state): 
        if np.random.random() <= self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.model.predict(state))
        
    def learn(self):
        if self.memory.stored < self.sample_size:
            return 
        state_sample, action_sample, reward_sample, next_state_sample, not_done_sample = self.memory.sample_memory(self.sample_size)
        
        batch_indexes = np.arange(self.sample_size)
        y_target = self.model.predict(state_sample)
        y_target[batch_indexes,action_sample] = reward_sample + self.gamma * np.max(self.model.predict(next_state_sample))*not_done_sample

        self.model.fit(state_sample,y_target,batch_size = 32,verbose = 0)     #TODO: add callback



def createEnv(x_dim,y_dim,depth,seed,n_agents):
    
    # Observation builder
    tree_observation = TreeObsForRailEnv(max_depth=depth)

    # Setup the environment
    env = RailEnv(
        width=x_dim,
        height=y_dim,
        rail_generator=sparse_rail_generator(
            seed=seed,
            grid_mode=False, 
        ),
        schedule_generator=sparse_schedule_generator(),
        number_of_agents=n_agents,
        obs_builder_object=tree_observation
    )

    return env 


def loop(n_episodes,max_steps,agent,env,isTraining,learn_every):

    for n in range(n_episodes):

        # Initialize episode
        steps = 0
        all_done = False

        # Reset environment
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        while not all_done and steps < max_steps:

            actions = {}
            for handle in env.get_agent_handles():
                if info['action_required'][agent]:
                    actions[handle] = agent.choose_action(obs[handle])
                else :
                    actions[handle] = 0
            
            # Environment step
            next_obs, all_rewards, done, info = env.step(actions)

            if isTraining:
                # Update replay buffer and train agent
                for handle in env.get_agent_handles():
                    # Only update the values when we are done or when an action was taken and thus relevant information is present
                    if actions[handle]!=0 or done[agent]:
                        # Add state to memory
                        agent.remember(
                            state = obs[handle],
                            action = actions[handle],
                            reward = all_rewards[handle],
                            new_state = next_obs[handle],
                            done = done[handle])

                # Learn
                if (steps + 1) % learn_every == 0:
                    agent.learn()
            else:
                env.render()
            

            # Update states        
            obs = next_obs
            # Are we done
            all_done = done['__all__']
            steps += 1

class Parameters():
    # General paramenters
    seed = 42
    
    # Observation parameters
    observation_tree_depth = 2

    # Environment parameters
    n_agents = 1
    x_dim = 25
    y_dim = 25
    state_size = 5 * sum([np.power(4,i) for i in range(observation_tree_depth+1)])
    action_size = 5

    # Agent parameters
    eps_start = 1.0
    eps_min = 0.01
    eps_decay = 0.997  
    gamma = 0.99
    lr = 0.5e-4
    sample_size = 512

    # Loop parameneters
    is_training = True
    n_episodes = 500
    max_steps = int(4 * 2 * (x_dim + y_dim + (n_agents / 5)))              #num cities
    learn_every = 10

def main():

    par = Parameters()

    env = createEnv(
        par.x_dim,
        par.y_dim,
        par.observation_tree_depth,
        par.seed,
        par.n_agents)

    agent = Agent(
        par.lr,
        par.gamma,
        par.action_size,
        par.eps_start,
        par.eps_min,
        par.sample_size,
        par.state_size,
        par.eps_decay)


    loop(par.n_episodes,par.max_steps,agent,env,par.is_training,par.learn_every)


        


