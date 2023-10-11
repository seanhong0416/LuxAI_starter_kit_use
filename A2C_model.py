
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np

lr_actor = 0.01
lr_critic = 0.01
gamma = 0.99
episodes_per_update = 1
num_of_actor_additional_condition = 6
num_of_critic_additional_condition = 4
num_of_actions = 13

class A2C_Net(nn.module):
    def __init__(self):
        super().__init__()
        
        self.mutual_conv = nn.Sequential(
            #input shape:64*64*6
            nn.Conv2d(6,40,kernel_size=5,stride=1),
            nn.ReLU(),
            #input shape:60*60*40
            nn.MaxPool2d(kernel_size=2),
            #input shape:15*15*40
            nn.Conv2d(40,40,kernel_size=5,stride=1),
            nn.ReLU(),
            #input shape:11*11*40
            nn.Flatten(),
            nn.Linear(11*11*40,1000),
            nn.ReLU(),
        )

        self.actor = nn.sequential(
            nn.Linear(1000+num_of_actor_additional_condition,100),
            nn.ReLU(),
            nn.Linear(100,num_of_actions),
            F.softmax(dim=-1),
        )

        self.critic = nn.Sequential(
            nn.Linear(1000+num_of_critic_additional_condition,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.ReLU(),
            nn.Linear(10,1),
        )
    
    def forward(self, maps, actor_input, critic_input):
        conv_input = torch.from_numpy(maps)
        conv_output = self.mutual_conv(conv_input)

        actions = {}
        for unit_id, unit_data in actor_input.items():
            actor_net_input = torch.cat((conv_output, torch.from_numpy(unit_data)))
            probs = self.actor(actor_net_input)
            probs = F.softmax(probs)
            actions[unit_id] = Categorical(probs)

        critic_net_input = torch.cat((conv_output, torch.from_numpy(critic_input)))
        value = self.critic(critic_net_input)

        return actions, value
    
    
    