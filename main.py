import torch
import matplotlib.pyplot as plt
import torch
from dqn_module import DQNagent
from game_env import Environment
from utils.helper import store_replay
from utils.helper import plot_durations

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    device = "cuda"
    current_game = True
    BATCH_SIZE = 512
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    n_actions = 5
    episode_durations = []
    rewardsss = []
    steps_done = 0
    episodes = 10000
    save_history = 0

    #state = env.reset()
    n_observations = 9

    dqn_agent = DQNagent(device, BATCH_SIZE, n_observations, n_actions, GAMMA, EPS_END, EPS_START, EPS_DECAY, LR, TAU)

    # TODO: fix this
    #target_net.load_state_dict(policy_net.state_dict())

    env = Environment("ai")
    env.reset()

    for i in range(0, episodes):

        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        print("RESETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")

        counter = 0
        max_reward = 0
        save_history += 1

        while current_game:
            counter += 1
            action = dqn_agent.select_action(state, env)

            observation, reward, terminated, truncated = env.step(action.item())
            max_reward += reward

            if max_reward == 0 and counter == 100 or counter == 1400:
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
                print(i, " ", max_reward, counter)
                episode_durations.append(counter)
                #TODO: Do i need this?
                is_ipython = 'inline' in plt.get_backend()
                if is_ipython:
                    from IPython import display
                
                plot_durations(is_ipython, episode_durations)
                break

        if save_history == 50 or max_reward >= 40:
            save_history = 0
            store_replay(env.history)