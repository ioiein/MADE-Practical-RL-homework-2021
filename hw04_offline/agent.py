import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.FloatTensor(np.array(state))
        with torch.no_grad():
            action = self.model.get_action(state)
        return action.detach().numpy()

    def reset(self):
        pass

