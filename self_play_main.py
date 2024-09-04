import pygame
import pygame.freetype
from game_env import Environment

if __name__ == "__main__":

    MODE = "player"
    env = Environment(MODE)

    current_game = True

    rewardd = 0

    while current_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                current_game = False     
                break
        
        keys = pygame.key.get_pressed()
        distance, reward, z, zzz = env.step(keys)
        rewardd += reward
        print(reward)