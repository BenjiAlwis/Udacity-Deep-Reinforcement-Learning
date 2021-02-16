import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, leak=0.01, seed=0, bn_mode=0):
        """ Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in hidden layers
            leak: amount of leakiness in leaky relu
        """
        super(Actor, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)

	# Dense layers
        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, action_size)

        self.bn = nn.BatchNorm1d(state_size)
	# Normalization layers
        self.bn1 = nn.BatchNorm1d(fc1)
        if bn_mode!=2:
            self.bn2 = nn.BatchNorm1d(fc2)     
        if bn_mode==3:    
            self.bn3 = nn.BatchNorm1d(action_size)   
        self.bn_mode=bn_mode
	####
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fc1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        
	
        if state.dim() == 1:
            state = torch.unsqueeze(state,0)
        
        if self.bn_mode==0:
	    # Batch Normalization of the input vector
            state = self.bn(state)
            x = F.leaky_relu(self.fc1(state), negative_slope=self.leak)
            x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
            return F.tanh(self.fc3(x))
        elif self.bn_mode==1:
            # Batch Normalization before Activation
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            return F.tanh(x)
        elif self.bn_mode==2:
            # Batch Normalization after Activation  
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        elif self.bn_mode==3:
            # Batch Normalization before Activation (alternate version)
            x = self.fc1(state)
            x = self.bn1(x)   
            x = F.relu(x)
            x = self.fc2(x)
            x = self.bn2(x)   
            x = F.relu(x)
            x = self.fc3(x)
            x = self.bn3(x)   
            return F.tanh(x)
        elif self.bn_mode==4:
            # Batch Normalization after Activation  (alternate version)
            x = F.relu(self.fc1(state))
            x = self.bn1(x) 
            x = F.relu(self.fc2(x))
            x = self.bn2(x)   
            return F.tanh(self.fc3(x)) 

class Critic(nn.Module):
    def __init__(self, state_size, action_size, fc1=256, fc2=128, fc3=128, leak=0.01, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            hidden_size:
        """
        super(Critic, self).__init__()
        self.leak = leak
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ Initilaize the weights using He et al (2015) weights """
        torch.nn.init.kaiming_normal_(self.fcs1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self.bn(state)
        x = F.leaky_relu(self.fcs1(state), negative_slope=self.leak)
        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x), negative_slope=self.leak)
        x = F.leaky_relu(self.fc3(x), negative_slope=self.leak)
        x =  self.fc4(x)
        return x
