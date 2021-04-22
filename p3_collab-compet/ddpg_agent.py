import numpy as np
import random
import copy
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.995 	        # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # L2 weight decay
MAX_LAYERS = 4


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
 """Deep Deterministic Policy Gradient Agent """
    
 def __init__(self, state_dim, action_dim, num_agents,seed=0, hidden_layers=[400,300,300],leak=0.1,bn_mode=0,approach=0):
  """Initialisation function

  Params
  ======
  state_dim: width of the state vector
  action_dim: width of the action vector
  num_agents: mumber of ddpg agents
  seed: random seed
  hidden_layers: widths of the hidden layers of the network
  leak: relu leak
  bn_mode: batch normalisation method
  approach (int): 0- state+actions at the input layer 1 - at the first hidden layer 
  
  """
  self.state_vector_dim = state_dim
  self.action_vector_dim = action_dim
  self.hiddenLayers = hidden_layers
  self.seed = seed
  self.leak = leak
  self.bn_mode = bn_mode
  self.approach = approach
  self.num_agents = num_agents
  self.actor_models = []#0-local, 1-target
  self.setup_actor_models()
  if len(self.hiddenLayers) < MAX_LAYERS:
   self.hiddenLayers.insert(0,self.state_vector_dim*num_agents)
  self.critic_models = []#0-local, 1-target
  self.setup_critic_models()
  

 def setup_actor_models(self):
  """ Create the actor (policy) networks """
  for i in range(2):
   self.actor_models.append(Actor(self.state_vector_dim, self.action_vector_dim,self.seed, self.hiddenLayers,self.leak,self.bn_mode).to(device))
  
  self.actor_optimizer = optim.Adam(self.actor_models[0].parameters(), lr=LR_ACTOR)
  for target, local in zip(self.actor_models[1].parameters(), self.actor_models[0].parameters()):
   target.data.copy_(local.data)

 def setup_critic_models(self):
  """ Create the critic networks """
  for i in range(2):
   self.critic_models.append(Critic(self.action_vector_dim*self.num_agents,self.seed, self.hiddenLayers,self.approach,self.leak,self.bn_mode).to(device))
  

  self.critic_optimizer = optim.Adam(self.critic_models[0].parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
  for target, local in zip(self.critic_models[1].parameters(), self.critic_models[0].parameters()):
   target.data.copy_(local.data)

 def soft_update(self, local_network, target_network, tau):
  """Update of target network weights using local network weights.
  Params
  ======
  local_network: local neural net
  target_network: target neural net
  tau (float): weight for combining 
  """
  #for tp, lp in zip(target_network.parameters(), local_network.parameters()):
   #tp.data.copy_(tau*lp.data + (1.0-tau)*tp.data)
  for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
   target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
 
 def learn(self,states,b_states,o_states,actions,b_actions,rwds,next_states,b_next_states,o_next_states,dns):
   """DDPG Learning for the multi-agent environment    
   Params
   ======   
   states - states vector of the agent
   b_states - concatonated states
   o_states - states vector of the other agent
   actions - actions vector of the agent
   b_actions - concatonated actions
   rwds - rewards for the agent
   next_states - next states vector of the agent
   b_next_states - concatonated next states
   o_next_states - next states vector of the other agent
   """
   nxt_acts = self.actor_models[1](next_states)
   Qt_nxt = self.critic_models[1](b_next_states, torch.cat((nxt_acts,self.actor_models[1](o_next_states)), dim=1).to(device))
   Qt = rwds + (GAMMA * Qt_nxt * (1 - dns))
   Qt_exp = self.critic_models[0](b_states, b_actions)
   critic_loss = F.mse_loss(Qt_exp, Qt)
   self.critic_optimizer.zero_grad()
   critic_loss.backward()
   self.critic_optimizer.step()
   predicted_actions = self.actor_models[0](states)
   actor_loss = -self.critic_models[0](b_states, torch.cat((predicted_actions,self.actor_models[0](o_states)),dim = 1).to(device)).mean()
   self.actor_optimizer.zero_grad()
   actor_loss.backward()
   self.actor_optimizer.step()
   self.soft_update(self.critic_models[0], self.critic_models[1], TAU)
   self.soft_update(self.actor_models[0], self.actor_models[1], TAU)      

 def save(self,num):
  """Save training progress for all agents"""
  for it in range(2):
   self.actor_models[it].save(it,num)
   self.critic_models[it].save(it,num)
  


  
