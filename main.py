import torch
import matplotlib.pyplot as plt
import torch
import itertools
import wandb
from dqn_module import DQNagent
from game_env import Environment
from utils.helper import calc_mean

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    device = "cuda"

    # we start running here:
    BATCH_SIZE = [256]
    GAMMA = [0.99]
    TAU = [0.005]
    LR = [1e-3]
    EPISODES = [6000]
    EPS_START = [0.9]
    EPS_END = [0.1]
    EPS_DECAY = [1000]
    N_ACTIONS = [4]
    N_OBSERVATIONS = [9]
    MAP = "track 1"
    ARCHITECTURE = "2 layer 64"
    LOSS_FUNCTION = ""
    LOCATION = "random"
    START_POS = "random"

    #TODO: add network architecture
    training_params = itertools.product(BATCH_SIZE, 
                                        GAMMA,
                                        EPS_DECAY, 
                                        TAU, 
                                        LR, 
                                        EPISODES, 
                                        EPS_START, 
                                        EPS_END, 
                                        EPS_DECAY, 
                                        N_ACTIONS, 
                                        N_OBSERVATIONS)
    number_of_exp = 0

    #TODO: Check if we log everything

    for batch_size, gamma, eps_decay, tau, lr, episodes, eps_start, eps_end, eps_decay, n_actions, n_observations in training_params:
        
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
                    "network_architecture": ARCHITECTURE,
                    "loss_function": LOSS_FUNCTION,
                    "map": MAP,
                    "location": LOCATION
                    # TODO: add info about the network
                    # Loss function maybe, add that ass param to dqn?
                    }
        )

        number_of_exp += 1
        current_game = True
        all_rewards = []

        dqn_agent = DQNagent(device, batch_size, n_observations, n_actions, gamma, eps_end, eps_start, eps_decay, lr, tau)
        env = Environment("ai", MAP, START_POS, 1)

        for i in range(0, episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            state = state[0]
            #TODO: check code below
            counter = 0
            max_reward = 0

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
                    wandb.log({"reward": max_reward, "epi_durations": counter, "mean_score": mean})
                    break
            
            #TODO: this could be wrong
            #TODO: When should I save?
            #TODO: CHECK LOSS?
            if max_reward >= 180 or mean >= 40:
                save_name = "track1" + str(max_reward) + "_" + str(mean)
                dqn_agent.save(save_name)
        wandb.finish()