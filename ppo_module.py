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
    def __init__(self, memory, device, n_observations, n_actions, gamma, gae_lambda, batch_size, mini_batchsize, update_epoch, clip_coef, learning_rate):
        super().__init__()
        #TODO: Maybe remove device?
        #self.critic = PPO(n_observations, 1, 1.0).to(device)
        #self.actor = PPO(n_observations, n_actions, 0.01).to(device)
        self.memory = memory
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = mini_batchsize
        self.update_epochs = update_epoch
        self.clip_coef = clip_coef
        #TODO: CLEAN THIS UP
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_observations, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_observations, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
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

        # bootstrap value if not done
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

        # flatten the batch
        b_obs = self.memory.obs.reshape((-1,) + (9,))
        b_logprobs = self.memory.logprobs.reshape(-1)
        b_actions = self.memory.actions.reshape((-1,) + ())
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.memory.values.reshape(-1)

        # Optimizing the policy and value network
        #TODO: PUT THIS CODE IN THE NETWORK
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # Can save old kl if we want
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                self.optimize(loss, max_grad_norm)

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    def save(self, name):
        #TODO maybe give string for name
        torch.save(self.actor.state_dict(), "saved_models/ppo_64_static_" + str(name) + ".pth")

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