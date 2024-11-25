import pygame
import pygame.freetype
import random
import ast
import numpy as np
from level import Level
from car import Car

"""
    This contains the game environment for a simple car racing game.
    The graphics are seperated from the game logic.
    TODO: Write docstring
		- Still sucks i removed old results
- Better way to save params, add start pos info and network info etc.
- Check self play + reward etc
- Clean run inference
"""

class Environment():
    """load and update the game, take in actions, keep score"""
    def __init__(self, mode: str, map: str, start_pos: str, number_of_players: int):
        self.action_space = np.array([0, 1, 2, 3])
        self.cars = []
        self.reward = []
        self.mode = mode
        self.map = map
        self.checkpoint_counter = 0
        self.last_checkpoint = []
        self.start_pos = start_pos
        self.walls = self._get_walls()
        self.checkpoints = self._set_checkpoints()
        self.number_of_players = number_of_players

        if mode != "ai":
            self._initialize_pygame()
        self.reset()

    def _initialize_pygame(self):
        """Initialize pygame components."""
        pygame.init()
        pygame.display.set_caption("Car sim :)")
        self.window = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE)
        self.background = pygame.image.load("assets/background2.png")
        self.car_group = pygame.sprite.Group()
        self.track_group = self._load_obstacles()

    def reset(self) -> None:
        self.cars = []
        for i in range(0, self.number_of_players):
            if self.start_pos == "random":
                # load start_pos
                # use random mode to select position
                # can drive around in the game to come up with different random points + headings
                # If heading is give, use that but default is 0
                pos = random.choice([(550, 125, 0), 
                                     (1562, 291, 1.46), 
                                     (1514, 591, 1.78), 
                                     (1531, 860, 1.36), 
                                     (1168, 956, 3.24), 
                                     (610, 942, 3.246), 
                                     (239, 681, 4.39), 
                                     (236, 459, 4.92), 
                                     (223, 279, 4.9), 
                                     (363, 129, 5.759)])
            else:
                pos = (550, 125, 0)
            car = Car("assets/car12.png", pos[0], pos[1], pos[2], self.mode)
            #car = Car("assets/car12.png", 550, 125, self.mode)
            self.cars.append(car)
            if self.mode == "player":
                self.car_group.add(car)

            car.update(self.walls)
            self.checkpoints = self._set_checkpoints()
            self.checkpoint_counter = 0
        return car.state
    
    def sample(self):
        return random.choice(self.action_space)

    def step(self, all_keys: list) -> None:

        #TODO: Shouldn't we get a reward for not crashing and just advancing in the right direction>?
        if self.mode != "ai":
            self.render()

        return_per_car = []
        for i, carr in enumerate(self.cars):
            carr.action(all_keys[i])
            reward = 0
            carr.update(self.walls)
            hit_wall_check = False
            finished = False

            if carr.hitwall is True:
                reward -= 1
                #carr.reset()
                hit_wall_check = True

            # TODO: check if there is a cleaner way of doing this.

            all_checkpoints_checked = []

            # We don't check score when doing inference
            if self.mode == "ai" or self.mode == "player":
                for i in self.checkpoints:
                    #print(i)
                    all_checkpoints_checked.append(carr.check_checkpoint(i))

                if True in all_checkpoints_checked:
                    find_index = all_checkpoints_checked.index(True)

                    if self.last_checkpoint != self.checkpoints[find_index]:
                        reward += 1
                        self.last_checkpoint = self.checkpoints[find_index]

                #TODO: CHeck if we need something like this
                #if self.checkpoint_counter == len(self.checkpoints):
                #   reward += 40
                #  self.checkpoint_counter = 0
            return_per_car.append([carr.state, reward, hit_wall_check, finished])
            if hit_wall_check is True:
                carr.reset()

        return return_per_car
        
    def render(self) -> None:
        #TODO: this one seems extra
        self.window.blit(self.background, (0, 0))
        #elf.car_group.update()
        self.track_group.draw(self.window)
        self.car_group.draw(self.window)
        pygame.display.flip()
        return None

    # What is returned?
    def _load_obstacles(self) -> None:
        self.track = Level("assets/" + self.map + ".png", 1920//2, 1080//2)
        self.track_group = pygame.sprite.Group()
        self.track_group.add(self.track)
        return self.track_group
    
    def _set_checkpoints(self) -> list:
        track = "checkpoints/" + self.map + ".txt"
        with open(track, 'r') as file:
            lines = file.readlines()
            checkpoints = [ast.literal_eval(line.strip()) for line in lines]
        file.close()
        return checkpoints
    
    def _get_walls(self) -> np.array:
        track = "track_info\\" + self.map + ".csv"
        return np.genfromtxt(track, delimiter=',', dtype = bool).reshape(1920, 1080)