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
from ppo_module import Memory
#TODO: check all args with og paper

# Define your parameters
args = Args(
    batch_size=[5000],
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
    minibatch_size= 1250,
    num_iterations= 4000,
    seed= 1,
    num_steps= [5000],
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
    memory = Memory(num_steps, args.num_envs, device)
    agent = PPOagent(memory, device, 9, 4, gamma, args.gae_lambda, batch_size, args.minibatch_size, update_epochs, clip_coef, args.learning_rate).to(device)
    global_step = 0

    for iteration in range(1, args.num_iterations + 1):
        #TODO: CHANGE THIS, seperate method for it
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow
        all_rewards = 0

        next_obs = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)[0]
        next_done = torch.zeros(1).to(device)
        
        for step in range(0, num_steps):
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                #TODO: Check on how to resolve this
                #values[step] = value.flatten()

            #TODO: Do i need to change to list? Dont think so
            next_obs_1, reward, terminations, truncations = envs.step(action)[0]
            next_done = np.logical_or([terminations], [truncations])
            memory.update_values(step, next_obs, action, logprob, reward, torch.Tensor(next_done), value.flatten())
            
            next_obs, next_done = torch.Tensor(next_obs_1).to(device), torch.Tensor(next_done).to(device)
            all_rewards += reward
            done = terminations or truncations

            if done:
                print(f"global_step={iteration}", "Steps taken", step, all_rewards)
                break
        
        agent.optimise_networks(next_obs, next_done, num_steps, args.ent_coef, args.vf_coef, args.norm_adv, args.clip_vloss, args.max_grad_norm, args.target_kl)
        # TODO: return something from this