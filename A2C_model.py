
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory

import numpy as np

lr_actor = 0.01
lr_critic = 0.01
gamma = 0.99
episodes_per_update = 1
num_of_actor_additional_condition = 
num_of_critic_additional_condition = 
num_of_actions = 


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
            nn.Linear(1006,100),
            nn.Linear(100,)
        )