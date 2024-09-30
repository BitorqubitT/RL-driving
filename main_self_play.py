import pygame
import pygame.freetype
from game_env import Environment

"""
    Self play to test the game
"""

if __name__ == "__main__":

    MODE = "player"
    #TODO: fix start point
    MAP_NAMES = ["scuffed monza", "track 1"]
    START_POS = [[(),(),()], [(), (), ()]]
    MAP = MAP_NAMES[1]
    env = Environment(MODE, MAP, START_POS)

    current_game = True
    rewardd = 0

    #TODO: Change action in game_env.

    while current_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                current_game = False     
                break
        
        # Add the buttons here for selfplay
        
        keys = pygame.key.get_pressed()
        distance, reward, z, zzz = env.step(keys)
        rewardd += reward
        print(reward)