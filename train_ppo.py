# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import random
from game_env import Environment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import wandb
from dataclass_helper import Args
from ppo_module import PPOagent
#TODO: check all args with og paper

# Define your parameters
args = Args(
    batch_size=[256],
    gamma=[0.99],
    tau=[0.005],
    lr=[1e-3],
    episodes=[6000],
    eps_start=[0.9],
    eps_end=[0.1],
    eps_decay=[1000],
    n_actions=[4],
    n_observations=[9],
    game_map="track 1",
    architecture="2 layer 64",
    loss_function="",
    location="random",
    start_pos="static",
    minibatch_size= 500,
    num_iterations= 4000,
    seed= 1,
    num_steps= [4000],
    num_envs= 1,
    update_epochs= [4],
    clip_coef= [0.2],
    clip_vloss= True,
    learning_rate=2.5e-4,
    anneal_lr= True,
    gae_lambda= 0.95,
    norm_adv= True,
    ent_coef= 0.01,
    vf_coef= 0.5,
    max_grad_norm=0.5,
    target_kl= None
)

number_of_exp = 0

for combination in args.check_and_iterate_combinations():
    batch_size, gamma, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations, num_steps, update_epochs, clip_coef = combination
    wandb.init(
        project = "rl_64_2layer_input4_randomspos_track1",
        name = f"experiment_{number_of_exp}",
        config={"batch_size": batch_size,
                "gamma": gamma,
                "eps_decay": eps_decay,
                "tau": tau,
                "lr": lr,
                "episodes": episodes,
                "eps_start": eps_start,
                "eps_end": eps_end,
                "n_actions": n_actions,
                "n_observations": n_observations,
                "network_architecture": args.architecture,
                "loss_function": args.loss_function,
                "map": args.game_map,
                "location": args.location
                }
    )

if __name__ == "__main__":
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    #TODO: Actually i can run multipe envs i think, that is why i have number of players
    NUMBER_OF_PLAYERS = 1
    envs = Environment("ai", args.game_map, args.start_pos, NUMBER_OF_PLAYERS)
    #TODO: Put actions and  observations from args.
    agent = PPOagent(device, 9, 4).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    print(num_steps, args.num_envs)
    obs = torch.zeros((num_steps, args.num_envs) + (9,)).to(device)
    actions = torch.zeros((num_steps, args.num_envs) + ()).to(device)
    logprobs = torch.zeros((num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((num_steps, args.num_envs)).to(device)
    dones = torch.zeros((num_steps, args.num_envs)).to(device)
    values = torch.zeros((num_steps, args.num_envs)).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    #next_done = torch.zeros(1).to(device)
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        all_rewards = 0

        next_obs = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs = next_obs[0]
        next_done = torch.zeros(1).to(device)
        
        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            #TODO: Do i need to change to list? Dont think so
            action = action.tolist()
            next_obs, reward, terminations, truncations = envs.step(action)[0]
            next_done = np.logical_or([terminations], [truncations])
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            all_rewards += reward
            done = terminations or truncations

            if done:
                print(f"global_step={global_step}", step, all_rewards)
                break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (9,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + ())
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
