import math
import pygame
import pygame.freetype
from pygame.sprite import Sprite
import numpy as np
from utils.ui import play_level
from utils.helper import store_replay
from utils.helper import read_replay

"""
TODO:
# Implement a training loop for rl
# Reward and penalty
# How to deal with finish?

# Use DQN algorithm
# Use DQN with img or sensors data (setup both)

# Statistics: save drive, raceline, pos+speed, crashes (write them to csv?)

# Will use raytracing for wall detection
# This might not be necessary if i train on image only.

Low prio:
# Smaller car, better track
# Make sure its easy to change levels
# Clean code + PEP8
# Faster time training
# private _ or __
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

    def __init__(self, car_image:str, x: float, y: float, player_type):
        super().__init__()
        #TODO: clean this angle stuff
        
        self.min_angle = math.radians(1)
        self.start_pos = (x, y)
        self.reversing = False
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
        """ 
            Car is basically 4 points and we create a hitbox.
            We have a position x,y
            From this we calculate the 4 edge points based on size image
            Then we calculate 4 lines around the car
            This way we can always calculate if the car hits the walls
            put some of these in util
        """
        all_edges = []

        angle_to_corner = math.degrees(math.atan(29.5/64.0))
        for i in [0.0, 45.0, 180.0, 225.0]:
            i = angle_to_corner + math.radians(i)
            heading = self.heading + i
            x = round(self.position[0] + math.cos(heading) * 70.472)
            y = round(self.position[1] + math.sin(heading) * 70.472)
            all_edges.append((x,y))
        return all_edges

    def calculate_angle(self, source, target):
        velocity = (target[0] - source[0], target[1] - source[1])
        angle = math.atan2(velocity[1], velocity[0])
        return angle
    
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
            angle = self.calculate_angle(point_pair[0], point_pair[1])
            # Set range based on distance between points
            # We know this because of the size of the car.
            if distance == 0 or distance == 2:
                range_set = 30
            if distance == 1 or distance == 3:
                range_set = 128
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
        self.heading += math.radians(angle_degrees)
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
        if self.speed <= 10:
            self.speed += amount

    def _brake(self) -> None:
        # Add more realistic way of breaking
        self.speed /= 2
        if (abs(self.speed) < 0.1):
            self.speed = 0

    def _cast_ray(self, arr, angle_offset) -> None:
        x, y = 0, 0
        heading = self.heading + angle_offset
        for i in range(0, 800):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            # If we find a wall return x, y
            if arr[x, y]:
                return x, y

    def distance_to_walls(self, walls) -> list:    
        ray_angles = [0, 70, -70]
        distances = []
        all_position = []
        for i, angle in enumerate(ray_angles):
            x, y = self._cast_ray(walls, angle)
            all_position.append((x, y))
            # We use distance for the sensors
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            distances.append(round(distance_to_wall))
        return all_position, distances
    
    def check_checkpoint(self, checkpoints) -> bool:
        if math.dist(self.position, checkpoints[0]) <= 70.0:
            return True
        return False

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
        # change this functiono depending on player or pc?
        # if pc just give int value between 1->8 actions?
        if self.player_type == "player":
            if keys[pygame.K_UP]:
                self._accelerate(0.1)
            if keys[pygame.K_DOWN]:
                self._brake()
            if self.speed != 0:
                if keys[pygame.K_RIGHT]:
                    self._turn(1.0)
                if keys[pygame.K_LEFT]:
                    self._turn(-1.0)
        else:
            # TODO: Does it make sense to do both at the same time?
            if keys == 0:
                pass
            elif keys == 1:
                self._accelerate(0.1)
            elif keys == 8:
                self._accelerate(0.1)
                self._turn(1.0)
            elif keys == 7:
                self._accelerate(0.1)
                self._turn(-1.0)
            elif keys == 4:
                self._accelerate(0.1)
            elif keys == 5:
                self._accelerate(0.1)
                self._turn(1.0)
            elif keys == 6:
                self._accelerate(0.1)
                self._turn(-1.0)
            elif keys == 3:
                self._turn(1.0)
            elif keys == 2:
                self._turn(-1.0)
            pass

# create one superclass i think
class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 
# add move to player class maybe?

class Environment():
    """load and update the game, take in actions, keep score"""
    def __init__(self, mode):
        self.width, self.height = 1920, 1080
        # TODO: put these under mode?
        clock = pygame.time.Clock()
        clock.tick_busy_loop(60)
        #TODO: Which ones should be private
        self.action_space = None
        self.observation_space = None
        self.reward = 0
        self.history = []
        self.mode = mode
        self.walls = self._get_walls("track_1.csv")
        self.checkpoints = self._set_checkpoints()
        if mode != "ai":
            self.width, self.height = 1920, 1080
            pygame.init()
            pygame.display.set_caption("Car sim :)")
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
            self.background = pygame.image.load("assets/background2.png")
            self.car_group = pygame.sprite.Group()
            self.track_group = self._load_obstacles()
            self.finish_group = self._load_finish()
        self.reset()

    def reset(self) -> None:
        # TODO: Cleaner way to solve this?
        if self.mode == "player" or self.mode == "view":
            self.car = Car("assets/car1.png", 950, 100, self.mode)
            self.car_group.add(self.car)
        else:
            self.car = Car("assets/car1.png", 950, 100, "ai")

    def step(self, keys) -> None:
        self.car.action(keys)
        self.history.append([self.car.position, keys])

        if self.mode != "ai":
            self.render()
        
        # Get hitbox positions
        indexlist = self.car.calculate_hitboxes()
        indexlisttranspose = np.array(indexlist).T.tolist()
        
        # Create a list with map values at hitbox position.
        # AKA check if wall was found
        self.car.update()
        if True in self.walls[ tuple(indexlisttranspose)]:
            self.car.reset()
            #TODO: Add reward or other stats, maybe?
            self.history.append("reset")

        # Calculate reward
        if self.car.check_checkpoint(self.checkpoints):
            self.reward += 1
            self.checkpoints.pop(0)
        return None
        
    def render(self) -> None:
        self.car.distance_to_walls(self.walls)
        self.window.blit(self.background, (0, 0))
        self.car_group.update()
        self.track_group.draw(self.window)
        self.car_group.draw(self.window)
        self.finish_group.draw(self.window)
        positions, distances = self.car.distance_to_walls(self.walls)
        for i in self.car.calculate_hitboxes():
            self.window.set_at((i[0], i[1]), (255, 113, 113))
        play_level(self.window, distances[0], distances[1], distances[2], self.car.speed, 1, self.reward)
        pygame.display.flip()
        return None

    # What is returned?
    def _load_obstacles(self) -> None:
        self.track = Level("assets/track 2.png", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.track_group = pygame.sprite.Group()
        self.track_group.add(self.track)
        return self.track_group
    
    def _load_finish(self) -> list:
        self.finish = Level("assets/finish2.png", 870, 100)
        self.finish_group = pygame.sprite.Group()
        self.finish_group.add(self.finish)
        return self.finish_group
    
    def _set_checkpoints(self) -> list:
        # Just make a list and draw at this position to check
        checkpoints = [(1170, 100),
                    (1470, 180),
                    (1670, 340),
                    (1800, 600),
                    (1470, 900),
                    (1170, 980),
                    (800, 940),
                    (440, 880),
                    (150, 580),
                    (270, 280),
                    (500, 150)
                    ]
        return checkpoints
    
    def _set_finish(self) -> tuple:
        finish = (870, 100)
        return finish

    # easy way to get the position of the track walls
    def _get_walls(self, track) -> np.array:
        return np.genfromtxt(track, delimiter=',', dtype = bool).reshape(1920, 1080)

if __name__ == "__main__":

    # player, view, ai
    MODE = "view"
    all_replays = []
    x = Environment(MODE)

    if MODE == "ai":
        for i in range(0, 10000):
            x.step(0)
            x.step(0)
            x.step(1)
            x.step(0)
            x.step(0)
            x.step(0)
            x.step(1)
            x.step(3)
        
        all_replays.append(x.history)
        replay_per_run = store_replay(all_replays)
    
    elif MODE == "player":
        current_game = True
        while current_game:
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    current_game = False     
            keys = pygame.key.get_pressed()
            x.step(keys)

    elif MODE == "view":
        
        FILENAME = "replays/rpl0-20240717-201542.csv"
        coordinates, moves = read_replay(FILENAME)
        #TODO: use enumerate
        for i in range(0, len(moves)):
            x.step(moves[i])


