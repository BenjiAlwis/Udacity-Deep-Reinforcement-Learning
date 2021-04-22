[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"



# Project 3: Collaboration and Competition

### Introduction

In this project, the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment is explored.

![Trained Agent][image1]

In this project, we use this environment in a two agent coperative game. They both have to keep hiting the ball without dropping it since when an agent hits the ball over the net it receives a reward of +0.1 where as it recieves a reward of -0.01 when the ball hits the ground or goes out of bound.In other words, the goal of each agent is to keep the ball in play.

Eight variables corresponding to the position and velocity of the ball and racket makes up the observation space. Each agent has its own observation (i.e. local observation). The action space available to an agent consists of two actions corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic. The goal for the agents is to receive an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

- After each episode, the rewards that each agent received are added up (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. Then take the maximum of these 2 scores is taken.
- This yields a single **score** for each episode.

The goal is achieved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started
1. Setting up:
	A Python3.5+ interpreter and other dependencies will have to be installed:
	1.1. Install Anaconda©: https://conda.io/docs/user-guide/install/index.html#
	1.2. Start a conda environment using Python 3.5+	
	1.3. Install the following packages together with their dependencies:
		- numpy: https://scipy.org/install.html
		- matplotlib: https://matplotlib.org/users/installing.html
		- PyTorch: https://pytorch.org/get-started/locally/
		- Unity ML: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md

 
2. Download the environment, based on your os, from the following links.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
3. Place the project files into the DRLND GitHub repository, placed in the `p3_collab-compet/` folder, and unzip (or decompress) the files. 


### Instructions

Please follow the instructions in the Jupyter notebook named`Tennis_Project.ipynb` to get started with training your own agent!
Press `Ctrl + Enter` to execute code inside each of the cells. 


  

