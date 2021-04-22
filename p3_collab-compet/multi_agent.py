import numpy as np
import random
import copy
import math
from collections import deque
import torch
from ddpg_agent import Agent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 200                   # Batch sz for replay buffer
GAMMA = 0.995                      # Discount factor
TAU = 1e-3                         # For soft update of target parameters
BUFFER_SIZE = int(1e5)             # replay buffer size
HIDDEN_LAYERS = [400,300,300]      # width of the hidden layers
LEAK = 0.1                         # leak factor in LEAK_RELU
BACH_NORM = 0                      # batch normalisation
APPROACH = 0                       # layer selection on where batch normalisation is done
SD = 10

class MultiAgent():
 """The multi-agent class """

 def __init__(self, state_dim, action_dim, num_agents):
  """initialisation
        
  Params
  ======
  state_dim: width of the state vector
  action_dim: width of the action vector
  num_agents: number of ddpg agents
  """
        
  super(MultiAgent, self).__init__()
  np.random.seed(SD)
  torch.manual_seed(SD)
  self.action_size = action_dim
  self.agents = [ Agent(state_dim,action_dim,num_agents, SD, HIDDEN_LAYERS, LEAK, BACH_NORM, APPROACH) 
                       for i in range(num_agents) ]
        
  self.buffer = ReplayBuffer(BUFFER_SIZE)
  self.noise = np.zeros(action_dim)      
  self.seed = random.seed(SD)
                  
 def calc_noise(self):
  """Generate a noise sample."""
  self.noise = self.noise +  (0.2 * np.random.randn(self.action_size) - 0.15 * self.noise)
  return self.noise

 def act(self, states):
  """Calc action for a given state - gradient based policy"""
  actions = []
  
  for agent, state in zip(self.agents, states):
   state = torch.from_numpy(state).float().to(device)
   agent.actor_models[0].eval()
   with torch.no_grad():
    action = agent.actor_models[0](state).cpu().data.numpy()
   agent.actor_models[0].train()
   action += self.calc_noise() 
   action = np.clip(action, -1, 1)
   actions.append(action)
   
   
  return actions

 def step(self, states, actions, rewards, next_states, done):
  """Add SARS into replay buffer"""
  self.buffer.add(states, actions, rewards, next_states, done)
  self.learn()

 def learn(self):
  """Call reinforcement learning """
  val = []
  if self.buffer.get_size() > BATCH_SIZE:
   ops = self.buffer.get_sample(BATCH_SIZE)
   for i in range(8):
    val.append(torch.from_numpy(ops[i]).float().to(device))
   for i in range(8,10):
    val.append(torch.from_numpy((ops[i]).astype(np.uint8)).float().to(device))

   steps = 1
   for agent in self.agents:
    if steps==1:
     agent.learn(val[0],torch.cat((val[0], val[1]), dim=1).to(device), val[1],val[2],torch.cat((val[2], val[3]), dim=1).to(device),val[4],val[6],torch.cat((val[6], val[7]),dim=1).to(device),val[7],val[8])
    else:
     agent.learn(val[1],torch.cat((val[1], val[0]), dim=1).to(device), val[0],val[3],torch.cat((val[3], val[2]), dim=1).to(device),val[5],val[7],torch.cat((val[7], val[6]),dim=1).to(device),val[6],val[9])

    steps += 1
        
                
    
 def save(self):
  """save check points"""
  for num, agent in enumerate(self.agents):
   agent.save(num)


class ReplayBuffer:
 """ Replay Buffer to store SARS vals and allow random sampling """

 def __init__(self, buffer_sz):
  """ Instance initialiser 
  Params
  ======
  buffer_size: Max buffer size """

  self.buffer = deque(maxlen=buffer_sz)
   
 def add(self, states, actions, rwds,next_states, dns):
  """ add new SARS """
  new_list=(states[0],states[1], actions[0],actions[1] ,rwds[0],rwds[1], next_states[0],next_states[1],dns[0],dns[1])
  self.buffer.append(new_list)
  
 def get_size(self):
  return len(self.buffer)
  
 def get_sample(self,batch_sz):
  """ give out  a sample batch 
  Params
  ======
  batch_sz: Number samples to output """
  ops = []
  if batch_sz <= len(self.buffer):
   sample_index = np.random.choice(len(self.buffer),batch_sz,False)
   sample = [self.buffer[int(i)] for i in sample_index]
   for i in range(8):
    ops1 = np.vstack( [s[i] for s in sample] )
    ops.append(ops1)
   for i in range(8,10):
    ops1 = np.vstack( [s[i] for s in sample] ).astype(np.uint8)
    ops.append(ops1)
   return ops


