import torch
import matplotlib.pyplot as plt
import torch
import itertools
import wandb
from dqn_module import DQNagent
from game_env import Environment
from utils.helper import calc_mean

if __name__ == "__main__":

# TODO: can remove this?

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    #TODO: Propper way of experimenting
    # Different file with params?

    device = "cuda"

    # we start running here:

    BATCH_SIZE = [256]
    GAMMA = [0.99]
    TAU = [0.005]
    LR = [1e-3]
    EPISODES = [6000]
    #TODO: EPISODES ISNT USED RIGHT
    EPS_START = [0.9]
    EPS_END = [0.1]
    EPS_DECAY = [1000]
    N_ACTIONS = [4]
    N_OBSERVATIONS = [9]

    #TODO: add network architecture

    training_params = itertools.product(BATCH_SIZE, GAMMA, EPS_DECAY, TAU, LR, EPISODES, EPS_START, EPS_END, EPS_DECAY, N_ACTIONS, N_OBSERVATIONS)
    number_of_exp = 0

    for batch_size, gamma, eps_decay, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations in training_params:
        print(f"running experiment with =", batch_size, gamma, eps_decay, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations)

        wandb.init(
            project = "rl_32_1layer_input4",
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
                    "n_observations": n_observations
                    # TODO: add info about the network
                    # Loss function maybe, add that ass param to dqn?
                    }
        )

        number_of_exp += 1
        current_game = True
        #TODO: can i give highest reward to WB?
        rewardsss = []
        all_rewards = []
        save_history = 0
        steps_done = 0

        dqn_agent = DQNagent(device, batch_size, n_observations, n_actions, gamma, eps_end, eps_start, eps_decay, lr, tau)
        env = Environment("ai")
        env.reset()

        for i in range(0, episodes):

            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            counter = 0
            max_reward = 0

            while current_game:
                counter += 1
                action = dqn_agent.select_action(state, env, training=True)
                #print(action.item())
                observation, reward, terminated, truncated = env.step(action.item())
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
                    wandb.log({"reward": max_reward, "epi_durations": counter, "mean_score": mean})
                    break
            
            #TODO: this could be wrong
            if max_reward >= 180 or mean >= 60:
                save_name = str(max_reward) + "_" + str(mean)
                dqn_agent.save(save_name)
                #maybe store the replay aswell?
                

        wandb.finish()