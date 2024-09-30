import math
import pygame
import pygame.freetype
import random
import ast
from pygame.sprite import Sprite
import numpy as np
from utils.helper import rotate_point
from utils.helper import calculate_angle
from utils.ui import play_level
from utils.helper import is_point_on_line

"""
    This contains the game environment for a simple car racing game.
    The graphics are seperated from the game logic.
    TODO: Check how to write good intro, name the different classes?
"""

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Car(Sprite):
    """
        A class to represent a car.
    """

    def __init__(self, car_image:str, x: float, y: float, player_type):
        """
        Args:
            car_image (str): location of the car image
            x (float): starting x position
            y (float): starting y position
            player_type (_type_): check if we play as ai, player or viewer
        """
        super().__init__()
        self.min_angle = math.radians(1)
        self.start_pos = (x, y)
        self.heading   = 0
        self.speed     = 0
        self.velocity  = pygame.math.Vector2(0, 0)
        self.position  = pygame.math.Vector2(x, y)
        self.player_type = player_type
        if self.player_type != "ai":
            self.rot_img = self._load_rotated_images(car_image)
            self.image   = self.rot_img[0]
            self.rect    = self.image.get_rect()
            self.mask    = pygame.mask.from_surface(self.image)


    def create_car_logic(self):
        # TODO: clean this function
        """ 
            Car is basically 4 points and we create a hitbox.
            We have a position x,y
            From this we calculate the 4 edge points based on size image
            Then we calculate 4 lines around the car
            This way we can always calculate if the car hits the walls
            put some of these in util
        """
        # Given center position of the object
        x, y = self.position[0], self.position[1]  # Example coordinates
        width, height = 64, 30
        angle_degrees = self.heading  # Example angle in degrees

        # Convert angle to radians
        angle_radians = angle_degrees

        # Calculate half-width and half-height
        half_width = width / 2
        half_height = height / 2

        # Calculate the corners relative to the center
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (-half_width, half_height),
            (half_width, half_height)
        ]

        # Calculate the rotated corner positions
        rotated_corners = [rotate_point(px, py, angle_radians) for px, py in corners]

        # Translate the rotated corners back to the original center
        translated_corners = [(x + px, y + py) for px, py in rotated_corners]

        # Print the corner positions
        translated_corners = [translated_corners[1], translated_corners[3], translated_corners[2], translated_corners[0]]

        return translated_corners
    
    def calculate_hitboxes(self) -> list:
        # Front, front_T, back_T, back_b       
        corners = self.create_car_logic()
        hitboxes = [[corners[0], corners[1]],
                    [corners[1], corners[2]],
                    [corners[2], corners[3]],
                    [corners[3], corners[0]]]

        #maybe check distance between points.
        hitbox_points = []
        for distance, point_pair in enumerate(hitboxes):
            #TODO: REPLACE GARBAGE CODE
            angle = calculate_angle(point_pair[0], point_pair[1])
            # Set range based on distance between points
            # We know this because of the size of the car.
            if distance == 0 or distance == 2:
                # TODO:Set these to height of car
                range_set = 30
            if distance == 1 or distance == 3:
                # TODO:Set these to length of car
                range_set = 64
            for i in range(0, range_set):
                x = round(point_pair[0][0] + math.cos(angle) * i)
                y = round(point_pair[0][1] + math.sin(angle) * i)
                hitbox_points.append((x, y))
        return hitbox_points

    def _load_rotated_images(self, car_image: str) -> list:
        car_image = pygame.image.load(car_image).convert_alpha()
        rotated_images = []
        for i in range(360):
            rotated_image = pygame.transform.rotozoom(car_image, 360 - 90 - (i), 1)
            rotated_images.append(rotated_image)
        return rotated_images

    def _turn(self, angle_degrees) -> None:
        #TODO: check if rounding matters
        self.heading = self.heading + math.radians(angle_degrees)
        if self.player_type != "ai":
            image_index = int(self.heading / self.min_angle) % len(self.rot_img)
            if (self.image != self.rot_img[ image_index ]):
                x,y = self.rect.center
                self.image = self.rot_img[ image_index ]
                self.rect  = self.image.get_rect()
                self.rect.center = (x,y)
                # need to update mask or collision will use og image
                self.mask = pygame.mask.from_surface(self.image)

    def _accelerate(self, amount) -> None:
        # Add more realistic way of accelerating + a normal speed cap
        if self.speed <= 3:
            self.speed += amount

    def _brake(self) -> None:
        # Add more realistic way of breaking
        #TODO: check different speed stuff
        self.speed = self.speed - (self.speed * 0.2)
        if (abs(self.speed) < 0.1):
            self.speed = 0

    # TODO: make this faster
    def _cast_ray(self, arr, angle_offset) -> None:
        x, y = 0, 0
        heading = self.heading + angle_offset
        #print(heading, self.position)
        # TODO: find more efficient way to do this
        # And make sure the range is enough
        for i in range(0, 1600):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            #print("we hawt", x, y)
            # If we find a wall return x, y
            if arr[x, y]:
                return x, y

    def distance_to_walls(self, walls) -> list:
        ray_angles = [0, 1.570, 4.712, 0.524, 5.76, 3.142, 3.67, 2.618]
        distances = []
        all_position = []
        for i, angle in enumerate(ray_angles):
            x, y = self._cast_ray(walls, angle)
            all_position.append((x, y))
            # We use distance for the sensors
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            distances.append((1500 - distance_to_wall) / 1500)
        return distances, all_position
    
    def check_checkpoint(self, checkpoints) -> bool:
        #TODO: maybe remove thie function all together.
        return is_point_on_line(checkpoints[0][0], checkpoints[0][1], (round(self.position[0]), round(self.position[1])))

    def update(self) -> None:
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        # TODO: FIX THIS FOR HUMAN and AI
        if self.player_type != "ai":
            self.rect.center = (round(self.position[0]), round(self.position[1]))

    # Maybe remove this, because we restart the environment?
    def reset(self) -> None:
        #self.image = self.rot_img[0]
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)
        self.speed = 0
        self.heading = 0

    def action(self, keys) -> None:
        if self.player_type == "plaayer":
            if keys[pygame.K_UP]:
                self._accelerate(1.0)
            if keys[pygame.K_DOWN]:
                self._brake()
            if self.speed != 0:
                if keys[pygame.K_RIGHT]:
                    self._turn(6)
                if keys[pygame.K_LEFT]:
                    self._turn(-6)
        else:
            #if keys == 0:
              #  pass
            if keys == 0:
                self._accelerate(1.0)
            elif keys == 2:
                self._turn(6)
            elif keys == 3:
                self._brake()
            elif keys == 1:
                self._turn(-6)

class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 

class Environment():
    """load and update the game, take in actions, keep score"""
    #TODO: Give map with params
    def __init__(self, mode: str, map: str, start_pos: tuple):
        #TODO maybe change with height we never change these
        # Should we have them here?
        self.action_space = np.array([0, 1, 2, 3])
        self.observation_space = None
        self.reward = 0
        self.history = []
        self.mode = mode
        self.map = map 
        self.walls = self._get_walls()
        self.checkpoints = self._set_checkpoints()
        self.extracheckp = self._set_checkpoints()
        
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
        # TODO: Cleaner way to solve this?
        if self.mode == "player":
            # track 2 950, 100
            # track 1 750, 200
            self.car = Car("assets/car12.png", 550, 125, self.mode)
            self.car_group.add(self.car)
        else:
            self.car = Car("assets/car12.png", 550, 125, "ai")
        distances, xx = self.car.distance_to_walls(self.walls)
        self.checkpoints = self._set_checkpoints()
        self.extracheckp = self._set_checkpoints()
        self.reward = 0
        self.history = []
        #TODO: check this, pretty sure this is used for normalisation
        distances.append(self.car.speed/3)

        return distances

    def sample(self):
        return random.choice(self.action_space)

    def step(self, keys) -> None:
        #TODO: WE ADDED SPEED TO DISTNCES AS A REWARD, clean this up

        # Using reward like this is pretty weird
        # Shouldn't we get a reward for not crashing and just advancing in the right direction>?
        self.reward = 0
        # TODO: CLEAN THIS UP
        self.history.append([self.car.position[0], self.car.position[1], self.car.heading, self.car.speed, self.reward, keys])
        self.car.action(keys)
        if self.mode != "ai":
            self.render()
        
        # Get hitbox positions
        indexlist = self.car.calculate_hitboxes()
        indexlisttranspose = np.array(indexlist).T.tolist()

        #TODO: use ___ for unused var? 
        distances, xx = self.car.distance_to_walls(self.walls)
        # Create a list with map values at hitbox position.
        # AKA check if wall was found
        self.car.update()
        
        if True in self.walls[tuple(indexlisttranspose)]:
            # Add a penalty for hitting a wall
            self.reward -= 1
            self.car.reset()
            #TODO: Add reward or other stats, maybe?
            #TODO: remove the weird distance append
            distances.append(self.car.speed/3)
            return distances, self.reward, True, False

        if len(self.checkpoints) == 0:
            self.reward += 40
            self.checkpoints = self._set_checkpoints()

        # Calculate reward
        # TODO: Other way of reloading checkpoints
        if self.car.check_checkpoint(self.checkpoints):
            self.reward += 1
            self.checkpoints.pop(0)

        # TODO: make this accesible through the environment by adding it at init?
        
        distances.append(self.car.speed/3)
        return distances, self.reward, False, False
        
    def render(self) -> None:
        #self.clock = pygame.time.Clock()
        #TODO: this one seems extra
        self.window.blit(self.background, (0, 0))
        self.car_group.update()
        self.track_group.draw(self.window)
        self.car_group.draw(self.window)
        distances, xxx = self.car.distance_to_walls(self.walls)
        for i in self.car.calculate_hitboxes():
            self.window.set_at((i[0], i[1]), (255, 113, 113))
        play_level(self.window, distances[0], distances[1], distances[2], self.car.speed, 1, self.reward)

        for i in self.checkpoints:
            pygame.draw.line(self.window, (255, 0, 0), i[0], i[1])
        
        for i in xxx:
            pygame.draw.line(self.window, (255, 0, 0), self.car.position, i)

        #self.clock.tick(60)
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