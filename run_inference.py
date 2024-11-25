from game_env import Environment
import torch
from dqn_module import DQNagent
from game_env import Environment

"""
    Let AI play the game.
    Load pytorch model and correct params.
    Run the game.
"""

if __name__ == "__main__":

    device = "cuda"
    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 0.005
    LR = 1e-3
    EPISODES = 4000
    EPS_START = 0.9
    EPS_END = 0.1
    EPS_DECAY = 1000
    N_ACTIONS = 4
    N_OBSERVATIONS = 9
    current_game = True

    dqn_agent = DQNagent(device, BATCH_SIZE, N_OBSERVATIONS, N_ACTIONS, GAMMA, EPS_END, EPS_START, EPS_DECAY, LR, TAU)
    dqn_agent2 = DQNagent(device, BATCH_SIZE, N_OBSERVATIONS, N_ACTIONS, GAMMA, EPS_END, EPS_START, EPS_DECAY, LR, TAU)
    
    file_name = "saved_models/someusefulname_64_randomspawn_track151_54.12.pth"
    file_name2 = "saved_models/someusefulname_64_randomspawn_track151_54.12.pth"
    dqn_agent.load(file_name)
    dqn_agent2.load(file_name2)
    all_agents = [dqn_agent, dqn_agent2]

    MAP = "track 1"
    START_POS = "static"
    NUMBER_OF_PLAYERS = 2
    env = Environment("player", "track 1", START_POS, NUMBER_OF_PLAYERS)
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    apwojd = 0

    while current_game:

        #TODO: Clean this up
        if apwojd == 0:
            all_states = [state, state]
        apwojd = 1

        all_actions = []
        
        for i, agent in enumerate(all_agents):
            action = agent.select_action(all_states[i][0], env, False)
            all_actions.append(action.item())

        all_car_data = env.step(all_actions)

        all_states = []
        for car_data in all_car_data:
            
            observation, reward, terminated, truncated = car_data
            if terminated:
                current_game = False

            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            all_states.append(state)