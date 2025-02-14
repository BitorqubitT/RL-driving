"""
This script contains the main training loop for training the DQN agent.
"""
import torch
import wandb
from dqn_module import DQNagent
from game_env import Environment
from utils.helper import calc_mean
from utils.dataclass_helper import Args

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    device = "cpu"
    number_of_exp = 0

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
        game_map= ["track 3"],
        architecture="2 layer 64",
        loss_function="",
        start_pos= ["random"]
    )

    for combination in args.check_and_iterate_combinations():
        print(combination)
        batch_size, gamma, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations, game_map, start_pos = combination

        wandb.init(
            project = "rl_dqn_2*64_run_2",
            name = f"experiment_{game_map}_{start_pos}",
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
                    "map": game_map,
                    }
        )

        number_of_exp += 1
        current_game = True
        all_rewards = []
        NUMBER_OF_PLAYERS = 1

        dqn_agent = DQNagent(device, batch_size, n_observations, n_actions, gamma, eps_end, eps_start, eps_decay, lr, tau)
        env = Environment("ai", game_map, start_pos, NUMBER_OF_PLAYERS)

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

                if max_reward == 0 and counter == 300 or counter == 10000:
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
            
            if max_reward >= 180 or mean >= 80:
                save_name = str(max_reward) + "_" + str(mean)
                dqn_agent.save(save_name, game_map, start_pos)
        wandb.finish()