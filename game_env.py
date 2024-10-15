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

    --------------------------------------------------------------
- remove the car hitbox and car logic
- Use the raytracers to calculate if we hit a wall.
- For each angle calculate the distance to hit is. (this is based on car size)
    - If we go under this value then hit wall
- Check how we deal with checkpoints
- Get main.py working for training.
- Perform old training and try to get the same results as before.
	- Save and test these
	- If we manage to get this performance than save this
		- Still sucks i removed old results
- Better way to save params, add start pos info and network info etc.
- Check self play + reward etc
- Clean run inference

"""

class Environment():
    """load and update the game, take in actions, keep score"""
    def __init__(self, mode: str, map: str, start_pos: tuple, number_of_players: int):
        self.action_space = np.array([0, 1, 2, 3])
        self.cars = []
        self.reward = []
        self.mode = mode
        self.map = map
        self.checkpoint_counter = 0
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

        # track 2 950, 100
        # track 1 750, 200
    def reset(self) -> None:
        self.cars = []
        for i in range(0, self.number_of_players):
            # TODO: Starting pos, depends on:
            # Map, training style
            # Keep for now, but change this within this function, based on map load start pos or something.
            # Param with alternate spawns?
            car = Car("assets/car12.png", 550, 125, self.mode)
            self.cars.append(car)
            if self.mode == "player":
                self.car_group.add(car)

            car.update(self.walls)
            #self.checkpoints = self._set_checkpoints()
            #self.checkpoint_counter = 0

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

            # Create a list with map values at hitbox position.
            # AKA check if wall was found
            car_hitboxes = carr.calculate_hitboxes()
            car_hitboxes_transposed = np.array(car_hitboxes).T.tolist()
            # Get overlap between the positions of walls and the car hitboxes

            if True in self.walls[tuple(car_hitboxes_transposed)]:
                # Add a penalty for hitting a wall
                reward -= 1
                carr.reset()
                hit_wall_check = False

            if carr.check_checkpoint(self.checkpoints):
                #TODO: How to fix this?????
                #Dont use this during inference right>
                reward += 1
                self.checkpoint_counter += 1
                self.checkpoints.append(self.checkpoints[0])
                self.checkpoints.pop(0)
            
            if self.checkpoint_counter == len(self.checkpoints):
                reward += 40
                self.checkpoint_counter = 0

            return_per_car.append([carr.state, reward, hit_wall_check, finished])
        
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
    
    # easy way to get the position of the track walls
    def _get_walls(self) -> np.array:
        track = "track_info\\" + self.map + ".csv"
        return np.genfromtxt(track, delimiter=',', dtype = bool).reshape(1920, 1080)