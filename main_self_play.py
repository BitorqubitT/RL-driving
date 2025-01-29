"""
    Initializes the game environment and runs the main game loop.
    Use this script to test the driving, track, game environment.
"""
import pygame
import pygame.freetype
from game_env import Environment

if __name__ == "__main__":

    MODE = "player"
    MAP_NAMES = ["scuffed monza", "track 1"]
    MAP = "track 3"
    START_POS = "static"
    NUMBER_OF_PLAYERS = 1

    env = Environment(MODE, MAP, START_POS, NUMBER_OF_PLAYERS)
    current_game = True
    total_reward = 0

    while current_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                current_game = False     
                break
        
        # Add the buttons here for selfplay
        keys = pygame.key.get_pressed()
        keys = [keys]
        observation, reward, terminated, truncated = env.step(keys)[0]
        total_reward += reward
        