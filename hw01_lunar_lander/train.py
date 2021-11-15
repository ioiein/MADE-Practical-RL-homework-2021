from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from collections import deque, namedtuple
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 4096
TRANSITIONS = 750000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
DEQUE_MAX_LEN = 500000

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class DQN:
    def __init__(self, state_dim, action_dim):
        self.device = 'cpu'
        self.steps = 0  # Do not change
        self.model = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )  # Torch model
        self.target = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.update_target_network()
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.lr_scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=0.0003,
                                                  max_lr=0.01, step_size_up=50, mode='triangular2',
                                                  cycle_momentum=False)
        self.memory = deque(maxlen=DEQUE_MAX_LEN)
        self.batch_size = BATCH_SIZE

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        # pass
        self.memory.append(Transition(*transition))

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        # pass
        batch_sample = random.sample(self.memory, self.batch_size)
        return batch_sample
        
    def train_step(self, batch):
        # Use batch to update DQN's network.
        # pass
        samples = Transition(*zip(*batch))
        sample_state = torch.Tensor(samples.state).to(self.device)
        sample_action = torch.Tensor(samples.action).unsqueeze(1).to(self.device)
        sample_reward = torch.Tensor(samples.reward).unsqueeze(1).to(self.device)
        next_state_values = torch.Tensor(samples.next_state).to(self.device)
        state_action_values = self.model(sample_state).gather(1, sample_action.long())
        done_mask = torch.Tensor(samples.done)

        state_action_values_expected = (self.target(next_state_values).max(1)[0].detach().unsqueeze(1) *
                                        GAMMA) + sample_reward

        loss = F.smooth_l1_loss(state_action_values, state_action_values_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target.load_state_dict(copy.deepcopy(self.model.state_dict()))
        # pass

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        result = self.model(torch.FloatTensor(state).to(self.device))
        return result.argmax().item()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model, "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns

if __name__ == "__main__":
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
    
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, _ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            print(f"Step: {i+1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
            dqn.save()
