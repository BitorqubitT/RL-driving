import itertools
from typing import List, Generator
from dataclasses import dataclass
from dqn_module import DQNagent
from game_env import Environment
from utils.helper import calc_mean
from dataclass_helper import Args
import torch
import wandb

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    device = "cuda"
    number_of_exp = 0

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
        start_pos="random"
    )
    for combination in args.check_and_iterate_combinations():
        batch_size, gamma, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations = combination

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
                    # TODO: add info about the network
                    # Loss function maybe, add that ass param to dqn?
                    }
        )

        number_of_exp += 1
        current_game = True
        all_rewards = []

        dqn_agent = DQNagent(device, batch_size, n_observations, n_actions, gamma, eps_end, eps_start, eps_decay, lr, tau)
        env = Environment("ai", args.game_map, args.start_pos, 1)

        for i in range(0, episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state = state[0]
            counter = 0
            max_reward = 0
            
            #TODO: change current game to number of steps?
            while current_game:
                counter += 1
                action = dqn_agent.select_action(state, env, training=True)
                observation, reward, terminated, truncated = env.step([action.item()])[0]
                observation = observation[0]
                max_reward += reward

                if max_reward == 0 and counter == 100 or counter == 5000:
                    truncated = True
                
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
                dqn_agent.push_memory(state, action, next_state, reward)
                state = next_state
                dqn_agent.optimize_model()
                dqn_agent.update_param()

                if done:
                    all_rewards.append(max_reward)
                    mean = calc_mean(all_rewards)
                    print("reward", max_reward, "epi_durations", counter, "mean_score", mean)
                    wandb.log({"reward": max_reward, "epi_durations": counter, "mean_score": mean})
                    break
            
            if max_reward >= 180 or mean >= 40:
                save_name = "track1" + str(max_reward) + "_" + str(mean)
                dqn_agent.save(save_name)
        wandb.finish()