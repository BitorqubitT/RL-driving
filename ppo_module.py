# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOagent(nn.Module):
    def __init__(self, device, n_observations, n_actions):
        super().__init__()
        #TODO: Maybe remove device?
        #self.critic = PPO(n_observations, 1, 1.0).to(device)
        #self.actor = PPO(n_observations, n_actions, 0.01).to(device)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(9, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(9, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 4), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class Memory():
    #TODO: Do we create a memory object in PPOAGENT?
    def __init__(self, num_steps, num_envs, device):
        self.obs = torch.zeros((num_steps, num_envs) + (9,)).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + ()).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

    def update_values(self, step, obs, actions, logprobs, rewards, dones, values):
        self.obs[step] = obs
        self.actions[step] = actions
        self.logprobs[step] = logprobs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
    
    def get_values(self):
        return self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values