# Deep Q Learning (DQN): Navigation

### Introduction

In this repository you will find the implementation of the Navigation Project. It is the first project for the Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

### Project Summary 

In this project, we train an agent to navigate and collect yellow bananas in a large, square world. it contains both yellow and blue banana as shown below.
![Train an agent to navigate and collect bananas](images/banana.gif)


### Rewards:
The agent is given a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a blue banana.  The goal is to maximise the cumulative score. This can be achieved by collecting as many yellow bananas as possible while minimising the number of blue bananas. The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### State Space 
Has 37 dimensions. It contains the agents velocity, along with ray-based precpetion of objects around the agents foward direction.

### Actions 
the agent has to learn how to select the best action for its current state. The four alternative discrete actions are:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.



### Getting Started

#### Step 1: Clone the DRLND Repository
1. Configure your Python environment by following [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). These instructions can be found in the [Readme.md](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md)
1. By following the instructions you will have installed PyTorch, the ML-Agents toolkits, and all the Python packages required to complete the project.
1. (For Windows users) The ML-Agents toolkit supports Windows 10. It has not been tested on older version but it may work.

#### Step 2: Download the Unity Environment 
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this folder, unzip (or decompress) the file and then write the correct path in the argument for creating the environment under the notebook `Navigation_solution.ipynb`:

``env = UnityEnvironment(file_name="Banana.app")
  env = UnityEnvironment(file_name="Banana.x86_64") - for Linux users
``
#### Step 3: Explore the Environment
After you have followed the instructions above, open Navigation.ipynb (located in the p1_navigation/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.
    
#### (Optional) Build your Own Environment
For this project, you can use the pre-built the Unity environment.

If you are interested in learning to build your own Unity environments after completing the project, you are encouraged to follow the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md), which walk you through all of the details of building an environment from a Unity scene.

### Train an agent
Use the pre-bult environment or Build your own local environment.
Make necessary adjustements for the path to the UnityEnvironment in the code.
Follow the instructions in `Navigation_solution.ipynb` to get started 

### Test an agent
- **DQN**: To run the original DQN algorithm, use the checkpoint `dqn.pth` to load the trained model. Also, choose theparameter `update_type` as `dqn`.
- **Double DQN**: If you want to run the Double DQN algorithm, use the checkpoint `ddqn.pth` for loading the trained model. Choose the parameter `update_type` as `double_dqn`.

### Description

- `dqn_agent.py`: agent implementation 
- `model.py`: DQN function approximator neural network implementation
- `dqn.pth`: saved model weights generated by the original DQN model
- `ddqn.pth`: saved model weights generated by the Double DQN model
- `Navigation_solution.ipynb`: notebook containing entry point to the solution





