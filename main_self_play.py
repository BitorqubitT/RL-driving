import pygame
import pygame.freetype
from game_env import Environment
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
"""
    Self play to test the game
"""


@dataclass
class SpawnHolder:
    data: Dict[str, List[Tuple[int, int, float]]] = field(default_factory = dict)

    def add_data(self, key: str, value: List[Tuple[int, int, float]]):
        self.data[key] = value
    
    def get_data(self, key: str) -> List[Tuple[int, int, float]]:
                return self.data.get(key, [])

spawn_holder = SpawnHolder()

spawn_holder.add_data('track 1', [(550, 125, 0),
                                (1562, 291, 1.46),
                                (1514, 591, 1.78),
                                (1531, 860, 1.36),
                                (1168, 956, 3.24),
                                (610, 942, 3.246),
                                (239, 681, 4.39),
                                (236, 459, 4.92),
                                (223, 279, 4.9),
                                (363, 129, 5.759)])




if __name__ == "__main__":

    MODE = "player"
    #TODO: fix start point
    MAP_NAMES = ["scuffed monza", "track 1"]
    MAP = "scuffed monza"
    START_POS = "random"
    NUMBER_OF_PLAYERS = 1
    env = Environment(MODE, MAP, START_POS, NUMBER_OF_PLAYERS)

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
        keys = [keys]
        observation, reward, terminated, truncated = env.step(keys)[0]
        rewardd += reward
        
        print(rewardd)