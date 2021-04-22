import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




class Actor(nn.Module):
 """Actor Neural Network."""

 def __init__(self, state_dim, action_dim,seed=10, hw=[400,300,300], leak=0.1,bn_mode=0):
  """Initialisation function.
  Params
  ======
  state_dim (int): size of the state vector
  action_dim (int): size of the action vector
  seed: random seed 
  hw: widths of the hidden layers
  leak: relu leak
  bn_mode: batch normalisation method
    
  """
  super(Actor, self).__init__()
  self.seed = torch.manual_seed(seed)
  self.leak = leak
  self.bn_mode = bn_mode
  self.ip_layer = nn.Linear(state_dim, hw[0])
  self.num_hidden_layers = len(hw)
  self.hls = nn.ModuleList()
  for layer_num in range(self.num_hidden_layers-1):
   self.hls.append(nn.Linear(hw[layer_num],hw[layer_num+1]))
  self.op_layer = nn.Linear(hw[-1], action_dim)
  self.bn = nn.BatchNorm1d(state_dim)
  self.bn1 = nn.BatchNorm1d(hw[0])
  self.reset_parameters()
        

 def reset_parameters(self):
  """ reset the weights using Kaiming normalisation"""
  torch.nn.init.kaiming_normal_(self.ip_layer.weight.data, a=self.leak, mode='fan_in')
  for hl in self.hls:
   torch.nn.init.kaiming_normal_(hl.weight.data, a=self.leak, mode='fan_in')
  torch.nn.init.uniform_(self.op_layer.weight.data, -3e-3, 3e-3)


 def forward(self, state):
  """States to action mapping by the policy network"""
  
  if state.dim() == 1:
   state = torch.unsqueeze(state,0)
  if self.bn_mode==0:
   state = self.bn(state)
  state = torch.squeeze(state,0)
  x = F.leaky_relu(self.ip_layer(state), negative_slope=self.leak)
  if self.bn_mode==1:
   x = self.bn1(x)
  for hl in self.hls:
   x = F.leaky_relu(hl(x), negative_slope=self.leak)
  return F.tanh(self.op_layer(x))

 def save(self,it,num):
  if it==0:
   filename = 'results/checkpoint_actor_local_' + str(num) + '.pth'
  else:
   filename = 'results/checkpoint_actor_target_' + str(num) + '.pth'            
  torch.save(self.state_dict(), filename)

class Critic(nn.Module):
 """Critic Neural Network"""

 def __init__(self,action_dim, seed=10,lw=[400,300,300],approach=0,leak=0.1,bn_mode=0):
  """Initialisation function
  Params
  ======
  action_dim : Dimension of each action
  seed (int): Random seed
  lw (int): size of the neural net layers, starting from the input layer
  approach (int): 0- state+actions at the input layer 1 - at the first hidden layer 
  leak: relu leak
  bn_mode: batch normalisation method
  
  """
  super(Critic, self).__init__()
  self.seed = torch.manual_seed(seed)
  self.approach = approach
  self.leak = leak
  self.bn = nn.BatchNorm1d(lw[0])
  self.bn1 = nn.BatchNorm1d(lw[1])
  self.num_layers = len(lw)
  self.nls = nn.ModuleList()

  if self.approach==0:
   self.nls.append(nn.Linear(lw[0]+action_dim, lw[1]))
   for layer_num in range(1,self.num_layers-1):
    self.nls.append(nn.Linear(lw[layer_num],lw[layer_num+1]))
  else:
   self.nls.append(nn.Linear(lw[0], lw[1]))
   self.nls.append(nn.Linear(lw[1]+action_dim, lw[2]))
   for layer_num in range(2,self.num_layers-1):
    self.nls.append(nn.Linear(lw[layer_num],lw[layer_num+1]))

  self.op_layer = nn.Linear(lw[-1], 1)
  self.reset_parameters()
        

 def reset_parameters(self):
  """ reset the weights using Kaiming normalisation"""
  for nl in self.nls:
   torch.nn.init.kaiming_normal_(nl.weight.data, a=self.leak, mode='fan_in')
  torch.nn.init.uniform_(self.op_layer.weight.data, -3e-3, 3e-3)


 def forward(self, state, action):
  """calculation of Q values for given state and action values"""
  
  
  if state.dim() == 1:
   state = torch.unsqueeze(state,0)
  if self.approach==0:
   x = torch.cat((state, action.float()), dim=1)
   for nl in self.nls:
    x = F.leaky_relu(nl(x), negative_slope=self.leak)
   return self.op_layer(x)
  elif bn_mode==0:
   x = self.bn(state)
   x = F.leaky_relu(self.nls[0](state), negative_slope=self.leak)
   x = torch.cat((x, action.float()), dim=1)
   for ln in range(1,len(self.nls)):
    x = F.leaky_relu(nls[ln](x), negative_slope=self.leak)
   return self.op_layer(x)
  else:
   x = F.leaky_relu(self.nls[0](state), negative_slope=self.leak)
   x = torch.cat((x, action.float()), dim=1)
   for ln in range(1,len(self.nls)):
    x = F.leaky_relu(nls[ln](x), negative_slope=self.leak)
   return self.op_layer(x)
        
 def save(self,it,num):
  if it==0:
   filename = 'results/checkpoint_critic_local_' + str(num) + '.pth'
  else:
   filename = 'results/checkpoint_critic_target_' + str(num) + '.pth'            
  torch.save(self.state_dict(), filename)
       

   
