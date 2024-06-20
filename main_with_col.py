import pygame
import math
import pygame_gui
import pygame
import pygame.freetype
from pygame.sprite import Sprite
from pygame.rect import Rect
from enum import Enum
from pygame.sprite import RenderUpdates
import numpy as np
from utils.ui import play_level

"""
TODO:
# Finish enviroment class
# Reward function - calc distance on track and timer.
# Create a player class
# Clean car class
# Be able to spawn multiple cars
# Smaller car, better track
# Draw functions maybe that draws everything
# Make sure its easy to change levels
# Clean code + PEP8
# Statistics: save drive, raceline, pos+speed, crashes (write them to csv?)
# USE NEAT (need different setup)
# USE DQN with img as input and only sensors 
# Faster time
"""

# Window size
WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080
WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Car(Sprite):
    def __init__(self, car_image:str, x: float, y: float):
        super().__init__()
        self.rot_img   = []
        self.min_angle = 1 
        self.rot_img = self._load_rotated_images(car_image)
        self.min_angle = math.radians(self.min_angle)
        self.image       = self.rot_img[0]
        self.rect        = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        #self.rect.center = (x, y)
        self.reversing = False
        self.heading   = 0
        self.speed     = 0 
        self.velocity  = pygame.math.Vector2(0, 0)
        self.position  = pygame.math.Vector2(x, y)
        self.start_pos = (x, y)

    def _load_rotated_images(self, car_image: str) -> list:
        car_image = pygame.image.load(car_image).convert_alpha()
        rotated_images = []
        for i in range(360):
            rotated_image = pygame.transform.rotozoom(car_image, 360 - 90 - (i), 1)
            rotated_images.append(rotated_image)
        return rotated_images

    def _turn(self, angle_degrees) -> None:
        self.heading += math.radians(angle_degrees) 
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
        if self.speed <= 10:
                self.speed += amount
        else: 
            self.speed -= amount

    def _brake(self) -> None:
        # Add more realistic way of breaking
        self.speed /= 2
        if (abs(self.speed) < 0.1):
            self.speed = 0

    def _cast_ray(self) -> None:
        # Maybe cast rays from the car
        # Just give the walls.
        # Keep data in car
        return None

    def update(self) -> None:
        self.velocity.from_polar((self.speed, math.degrees(self.heading)))
        self.position += self.velocity
        self.rect.center = (round(self.position[0]), round(self.position[1]))

    # Maybe remove this?
    def reset(self) -> None:
        self.image = self.rot_img[0]
        self.position  = pygame.math.Vector2(self.start_pos[0], self.start_pos[1])
        self.velocity  = pygame.math.Vector2(0, 0)

    def action(self, input) -> None:
        #keys = pygame.key.get_pressed()
        if (event.key == pygame.K_UP):  
            self._accelerate(0.5)
        elif (event.key == pygame.K_DOWN):  
            self._brake()
        elif (event.key == pygame.K_LEFT):
            self._turn(-1.0)
        elif (event.key == pygame.K_RIGHT):
            self._turn(1.0)
        


# create one superclass i think
class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 
# add move to player class maybe?

# easy way to get the position of the track walls
def get_walls(track, width, height) -> np.array:
    all_walls = []
    for x in range(0, width):
        line = []
        for j in range(0, height):
            if track.get_at((x, j)):
                line.append(True)
            else:
                line.append(False)
        all_walls.append(line)
    wall_pos = np.array([np.array(x) for x in all_walls])
    return wall_pos

# We actually want the distance at some point
def cast_ray(car_pos, arr, heading, angle_offset):
    x, y = 0, 0
    heading = heading + angle_offset
    for i in range(0, 800):
        x = round(car_pos[0] + math.cos(heading) * i)
        y = round(car_pos[1] + math.sin(heading) * i)
        if arr[x, y]:
            return x, y

class Environment():
    """load and update the game, take in actions, keep score"""
    def __init__(self):
        pygame.init()
        self.fps = 120
        self.width, self.height = 1920, 1080
        pygame.display.set_caption("Car sim :)")
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
        self.background = pygame.image.load("assets/background2.png")
        self.track = Level("assets/track 2.png", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.finish = Level("assets/finish2.png", 870, 100)
        self.action_space = None
        self.observation_space = None
        self.reward = 0
        self.car_group = pygame.sprite.Group()
        self.track_group = pygame.sprite.Group()
        self.finish_group = pygame.sprite.Group()
        self.car_group.add(self.car_group)
        self.track_group.add(self.track)
        self.finish_group.add(self.finish)
        self.wall_pos = _load_walls()
        # other way of doing this

        self.reset()

    def reset(self) -> None:
        # Do we want to reset the environment?
        laps = 0

        self.car = Car("assets/car1.png", 950, 100)

        if pygame.sprite.spritecollide(self.track, self.car_group, False, pygame.sprite.collide_mask):
            self.car.reset()
        if pygame.sprite.spritecollide(self.finish, self.car_group, False, pygame.sprite.collide_mask):
            laps += 1


    """Do we need step?"""
    def step(self, action):

        return None
        # Check collision:

    def render(self) -> None:
        return None

    def load_obstacles() -> None:
        return None
    
    def _load_finish(self) -> list:
        return []





if __name__ == "__main__":

    clock = pygame.time.Clock()


    while not done:

        for event in pygame.event.get():
            if (event.type == pygame.QUIT):
                done = True
            elif (event.type == pygame.KEYUP):
                if (event.key == pygame.K_UP):  
                    black_car.accelerate(0.5)
                elif (event.key == pygame.K_DOWN):  
                    black_car.brake()
        
        keys = pygame.key.get_pressed()
        move(black_car, keys)

        window.blit(background, (0, 0))
        car_group.update()
        track_group.draw(window)
        car_group.draw(window)
        finish_group.draw(window)

        # Cast diff rays
        ray_angles = [0, 70, -70]
        distances = []
        for i, angle in enumerate(ray_angles):
            x, y = cast_ray(black_car.position, wall_pos, black_car.heading, angle)
            distance_to_wall = math.sqrt((x - black_car.position[0]) ** 2 + (y - black_car.position[1]) ** 2)
            distances.append(round(distance_to_wall))
            pygame.draw.line(window, (255, 113, 113), [black_car.position[0], black_car.position[1]], [x, y], 5)

        play_level(window, distances[0], distances[1], distances[2], black_car.speed, laps)

        pygame.display.flip()
        # set fps
        clock.tick_busy_loop(60)

    pygame.quit()


