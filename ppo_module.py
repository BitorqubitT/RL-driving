# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, TensorDataset

class PPOagent(nn.Module):
    def __init__(self, memory, device, n_observations, n_actions, gamma, gae_lambda, batch_size, mini_batchsize, update_epoch, clip_coef, learning_rate):
        super().__init__()
        #TODO: Maybe remove device?
        self.critic = PPO(n_observations, 1, 1.0).to(device)
        self.actor = PPO(n_observations, n_actions, 0.01).to(device)
        self.memory = memory
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = mini_batchsize
        self.update_epochs = update_epoch
        self.clip_coef = clip_coef
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate, eps=1e-5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def optimize(self, loss, max_grad_norm):
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

    def optimise_networks(self, next_obs, next_done, num_steps, ent_coef, vf_coef, norm_adv, clip_vloss, max_grad_norm, target_kl):
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.memory.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.memory.dones[t + 1]
                    nextvalues = self.memory.values[t + 1]
                delta = self.memory.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.memory.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.memory.values

        # Flatten the batch
        b_obs = self.memory.obs.reshape((-1,) + self.memory.obs.shape[2:])
        b_logprobs = self.memory.logprobs.reshape(-1)
        b_actions = self.memory.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.memory.values.reshape(-1)

        # Create DataLoader for batch processing
        dataset = TensorDataset(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=False)

        clipfracs = []
        for _ in range(self.update_epochs):
            for batch in dataloader:
                mb_obs, mb_logprobs, mb_actions, mb_advantages, mb_returns, mb_values = batch

                _, newlogprob, entropy, newvalue = self.get_action_and_value(mb_obs, mb_actions.long())
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(newvalue - mb_values, -self.clip_coef, self.clip_coef)
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                self.optimize(loss, max_grad_norm)

                if target_kl is not None and approx_kl > target_kl:
                    break

        #y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        #var_y = np.var(y_true)
        #explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y\
        #TODO: return some useful stats
        return None

    def save(self, name, game_map, start_pos):
        #TODO maybe give string for name
        torch.save(self.actor.state_dict(), "saved_models/ppo_" + game_map + "_" + start_pos + "_" + str(name) + ".pth")

    def load(self, file_name):
        #TODO: need a file with param to match the pth
        self.actor.load_state_dict(torch.load(file_name))
        self.actor.eval()

class Memory():
    #TODO: Do we create a memory object in PPOAGENT?
    """ 
        Class that holds memory for ppoagent
    """
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
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO(nn.Module):
    def __init__(self, n_observations, n_actions, std):
        super(PPO, self).__init__()
        self.layer1 = layer_init(nn.Linear(n_observations, 64))
        self.layer2 = layer_init(nn.Linear(64, 64))
        self.layer3 = layer_init(nn.Linear(64, n_actions), std=std)

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        return self.layer3(x)
