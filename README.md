# Flatland Challenge

<img src="flatland.gif" width="500">

This challenge tackles a key problem in the transportation world:  
**How to efficiently manage dense traffic on complex railway networks?**

This is a real-world problem faced by many transportation and logistics companies around the world such as the Swiss Federal Railways and Deutsche Bahn. The Flatland challenge aims to address the problem of train scheduling and rescheduling by providing a simple grid world environment and allowing for diverse experimental approaches.

The goal is to make all the trains arrive at their target destination with minimal travel time. In other words, we want to minimize the number of steps that it takes for each agent to reach its destination. In the simpler levels, the agents may achieve their goals using ad-hoc decisions. But as the difficulty increases, the agents have to be able to plan ahead.

## How to run

 ```console
   $ python train.py [-e / --episodes EPISODES] [-pa / --par_agn AGENT] [-pe / --par_env ENVIRONMENT] [-c / --checkpoint CHECKPOINT] [-t / --tag TAG]

   ```
* `EPISODES`: number of episodes to run
* `AGENT`: agent .json configuration file
* `ENVIRONMENT`: environment .json configuration file
* `CHECKPOINT`: checkpoint file 
* `TAG`: wandb tag of the run

## Package requirements

`python 3.6.13`

`flatland-rl`  
`gitpython`  
`numpy`  
`tensorflow`  
`tensorflow-probability`  
`tensorflow-addons`  
`wandb`  