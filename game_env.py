import pygame
import pygame.freetype
import random
import ast
import os
import numpy as np
from car import Car
from dataclass_helper import SpawnHolder
from pygame.sprite import Sprite

class Environment():
    """Load and update the game, take in actions, and keep score.

    Attributes:
        action_space (np.ndarray): The possible actions.
        cars (list): The list of car objects.
        reward (list): The list of rewards.
        mode (str): The mode of the game ('ai' or 'player').
        map (str): The map of the game.
        checkpoint_counter (int): The counter for checkpoints.
        last_checkpoint (list): The last checkpoint reached.
        start_locations (SpawnHolder): The start locations for the cars.
        start_pos (str): The starting position of the cars.
        walls (np.ndarray): The array representing the walls.
        checkpoints (list): The list of checkpoints.
        number_of_players (int): The number of players.
    """
    
    def __init__(self, mode: str, map: str, start_pos: str, number_of_players: int):
        """Initializes the Environment object.

        Args:
            mode (str): The mode of the game ('ai' or 'player').
            map (str): The map of the game.
            start_pos (str): The starting position of the cars.
            number_of_players (int): The number of players.
        """
        self.action_space = np.array([0, 1, 2, 3])
        self.cars = []
        self.reward = []
        self.mode = mode
        self.map = map
        self.checkpoint_counter = 0
        self.last_checkpoint = []
        self.start_locations = self._load_start_locations()
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

    def _load_start_locations(self):
        """Load the start locations for the cars.

        Returns:
            SpawnHolder: The start locations for the cars.
        """
        spawnpoints = SpawnHolder()
        if self.map == "all":
            for maps in os.listdir("/spawn_info"):
                maps = maps.replace(".csv", "")
                spawnpoints.load_data_from_file(maps)
        else:
            spawnpoints.load_data_from_file(self.map)
        return spawnpoints

    def reset(self) -> list:
        """Reset the environment to its initial state.

        Returns:
            list: The state of the car.
        """
        self.cars = []
        for _ in range(0, self.number_of_players):
            if self.start_pos == "random":
                pos = random.choice(self.start_locations.get_spawn_data(self.map))
            else:
                pos = self.start_locations.get_spawn_data(self.map)[0]
            #TODO: TRASH -> Clean this
            diff_car = "car12.png"
            diff_car2 = "car13.png"
            car_name = "assets/"
            if _ == 0:
                car_name += diff_car
            elif _ == 1:
                car_name += diff_car2
            car = Car(car_name, pos[0], pos[1], pos[2], self.mode)
            #car = Car("assets/car12.png", pos[0], pos[1], pos[2], self.mode)
            self.cars.append(car)
            if self.mode == "player":
                self.car_group.add(car)
            car.update(self.walls)
            self.checkpoints = self._set_checkpoints()
            self.checkpoint_counter = 0
        return car.state
    
    def sample(self):
        """Sample a random action from the action space.

        Returns:
            int: A random action.
        """
        return random.choice(self.action_space)

    def step(self, all_keys: list) -> list:
        """Take a step in the environment.

        Args:
            all_keys (list): The list of keys pressed for each car.

        Returns:
            list: The state, reward, hit wall check, and finished status for each car.
        """
        if self.mode != "ai":
            self._render()

        return_per_car = []
        for i, carr in enumerate(self.cars):
            carr.action(all_keys[i])
            reward = 0
            carr.update(self.walls)
            hit_wall_check = carr.hitwall
            finished = False
            if hit_wall_check:
                reward -= 1

            all_checkpoints_checked = [carr.check_checkpoint(checkpoint) for checkpoint in self.checkpoints]

            if self.mode in {"ai", "player"}:
                if any(all_checkpoints_checked):
                    find_index = all_checkpoints_checked.index(True)
                    if self.last_checkpoint != self.checkpoints[find_index]:
                        reward += 1
                        self.last_checkpoint = self.checkpoints[find_index]

                # Maybe extra reward for whole lap

            return_per_car.append([carr.state, reward, hit_wall_check, finished])
            if hit_wall_check:
                carr.reset()

        return return_per_car

    def _render(self) -> None:
        """Render the game environment."""
        self.window.blit(self.background, (0, 0))
        self.track_group.draw(self.window)
        self.car_group.draw(self.window)
        pygame.display.flip()
        return None

    def _load_obstacles(self) -> None:
        """Load the obstacles for the track.

        Returns:
            pygame.sprite.Group: The group of track obstacles.
        """
        self.track = Level("assets/" + self.map + ".png", 1920//2, 1080//2)
        self.track_group = pygame.sprite.Group()
        self.track_group.add(self.track)
        return self.track_group
    
    def _set_checkpoints(self) -> list:
        """Set the checkpoints for the track.

        Returns:
            list: The list of checkpoints.
        """
        track = "checkpoints/" + self.map + ".txt"
        with open(track, 'r') as file:
            lines = file.readlines()
            checkpoints = [ast.literal_eval(line.strip()) for line in lines]
        file.close()
        return checkpoints
    
    def _get_walls(self) -> np.array:
        """Get the walls for the track.

        Returns:
            np.ndarray: The array representing the walls.
        """
        track = "track_info\\" + self.map + ".csv"
        return np.genfromtxt(track, delimiter=',', dtype = bool).reshape(1920, 1080)

class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 