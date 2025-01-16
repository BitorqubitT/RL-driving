"""
    Load a map to extract wall positions.
"""
import pygame
import numpy as np
from game_env import *

def get_walls(track: pygame.Surface) -> np.array:
    """
    Gets the position of the walls on the track.

    Args:
        track (pygame.Surface): The track surface.

    Returns:
        np.ndarray: A 2D array representing the position of the walls.
    """
    all_walls = []
    for x in range(0, 1920):
        line = []
        for y in range(0, 1080):
            if track.get_at((x, y)):
                line.append(True)
            else:
                line.append(False)
        all_walls.append(line)
    wall_pos = np.array([np.array(x) for x in all_walls])
    return wall_pos

if __name__ == "__main__":

    window = pygame.display.set_mode((1920, 1080))
    pygame.init()
    map_name = "track 3"
    track = Level("assets/" + map_name + ".png", 1920//2, 1080//2)
    track_group = pygame.sprite.Group()
    track_group.add(track)
    new_track = get_walls(track.mask)
    new_track.tofile("track_info/" + map_name + ".csv", sep=',')