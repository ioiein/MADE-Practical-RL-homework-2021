import random
import numpy as np
import os
import torch

random.seed(25)
np.random.seed(25)
torch.manual_seed(25)


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action, _, _ = self.model.act(state)
            return action.cpu().numpy()

    def reset(self):
        pass

