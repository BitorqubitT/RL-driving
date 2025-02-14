import math
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from typing import Tuple, List, Optional

Transition = namedtuple('Transition',
        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    Replay Memory for storing transitions experienced by the agent.

    Attributes:
        memory (deque): A deque to store the transitions.
    """
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQNagent():

    def __init__(self, device: torch.device, batch_size: int, n_observations: int, n_actions: int, 
                gamma: float, eps_end: float, eps_start: float, decay: float, lr: float, tau: float): 
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.eps_decay = decay
        self.tau = tau
        self.steps_done = 0

    def push_memory(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor) -> None:
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state: torch.Tensor, env, training: bool = False) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold or not training:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.sample()]], device=self.device, dtype=torch.long)

    def update_param(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 10)
        self.optimizer.step()

    def save(self, name: str, game_map: str, start_pos: str):
        torch.save(self.policy_net.state_dict(), "saved_models/dqn_" + game_map + "_" + start_pos + "_" + str(name) + ".pth")

    def load(self, file_name: str):
        #TODO: need a file with param to match the pth
        self.policy_net.load_state_dict(torch.load(file_name))
        self.policy_net.eval()

class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

 