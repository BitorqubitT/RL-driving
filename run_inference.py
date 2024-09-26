import pygame.freetype
from game_env import Environment
import torch
import torch
from dqn_module import DQNagent
from game_env import Environment

"""
    Let AI play the game.
    Load pytorch model and correct params.
    Run the game.
"""

#TODO:
# Get the network action selection working

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
    file_name = "saved_models/someusefulname_311_21.83.pth"
    dqn_agent.load(file_name)
    

    env = Environment("player")
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while current_game:
        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        current_game = False     
        #        break
        
        action = dqn_agent.select_action(state, env, False)
        observation, reward, terminated, truncated = env.step(action.item())
        
        if terminated:
            current_game = False
        
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)