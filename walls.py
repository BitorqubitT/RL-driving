import pygame
import numpy as np
from game_env import *

"""
    This script gets the position of the walls on a track.

"""


window = pygame.display.set_mode((1920, 1080))

pygame.init()

map_name = "scuffed monza"

track = Level("assets/" + map_name + ".png", 1920//2, 1080//2)
track_group = pygame.sprite.Group()
track_group.add(track)

# easy way to get the position of the track walls
def get_walls(track) -> np.array:
    all_walls = []
    for x in range(0, 1920):
        line = []
        for j in range(0, 1080):
            if track.get_at((x, j)):
                line.append(True)
            else:
                line.append(False)
        all_walls.append(line)
    wall_pos = np.array([np.array(x) for x in all_walls])
    return wall_pos

new_track = get_walls(track.mask)
new_track.tofile("track_info/" + map_name + ".csv", sep=',')


