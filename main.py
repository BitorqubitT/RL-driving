import math
import pygame
import pygame.freetype
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pygame.sprite import Sprite
import numpy as np
from utils.ui import play_level
from utils.helper import store_replay
from utils.helper import read_replay
from dqn import DQN
from dqn import select_action
from dqn import plot_durations
from dqn import ReplayMemory
from collections import namedtuple, deque
import asyncio

"""
TODO:
# TORCH CUDA
# Might want to normalise the sensor data / scale them
# Reward and penalty
# Maybe make it possible to keep going after one lap
# Can also implement this after network

# Use DQN algorithm
# Use DQN with img or sensors data (setup both)

# Statistics: save drive, raceline, pos+speed, crashes (write them to csv?)

# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py 
# inspiratoin for commenting

Low prio:
# How to deal with finish?
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
        #print(heading, self.position)
        # TODO: find more efficient way to do this
        # And make sure the range is enough
        for i in range(0, 900):
            x = round(self.position[0] + math.cos(heading) * i)
            y = round(self.position[1] + math.sin(heading) * i)
            #print("we hawt", x, y)
            # If we find a wall return x, y
            if arr[x, y]:
                return x, y

    def distance_to_walls(self, walls) -> list:
        ray_angles = [0, 70, -70]
        distances = []
        #all_position = []
        for i, angle in enumerate(ray_angles):
            x, y = self._cast_ray(walls, angle)
            #all_position.append((x, y))
            # We use distance for the sensors
            distance_to_wall = math.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
            distances.append(round(distance_to_wall))
        return distances
    
    def check_checkpoint(self, checkpoints) -> bool:
        if math.dist(self.position, checkpoints[0]) <= 80.0:
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
        self.action_space = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
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
            # TODO: maybe add distance to walls here
        self.reset()

    def reset(self) -> None:
        # TODO: Cleaner way to solve this?
        if self.mode == "player" or self.mode == "view":
            self.car = Car("assets/car1.png", 950, 100, self.mode)
            self.car_group.add(self.car)
        else:
            self.car = Car("assets/car1.png", 950, 100, "ai")
        distances = self.car.distance_to_walls(self.walls)
        return distances

    def sample(self):
        return random.choice(self.action_space)

    def step(self, keys) -> None:
        self.car.action(keys)
        self.history.append([self.car.position, keys])

        if self.mode != "ai":
            self.render()
        
        # Get hitbox positions
        indexlist = self.car.calculate_hitboxes()
        indexlisttranspose = np.array(indexlist).T.tolist()
        
        distances = self.car.distance_to_walls(self.walls)
        
        # Create a list with map values at hitbox position.
        # AKA check if wall was found
        self.car.update()
        if True in self.walls[ tuple(indexlisttranspose)]:
            # Add a penalty for hitting a wall
            self.reward -= 1
            self.car.reset()
            #TODO: Add reward or other stats, maybe?
            self.history.append("reset")
            return distances, self.reward, True, False
        
        # Calculate reward
        if self.car.check_checkpoint(self.checkpoints):
            self.reward += 1
            self.checkpoints.pop(0)

        # TODO: make this accesible through the environment by adding it at init?

        return distances, self.reward, False, False
        
    def render(self) -> None:
        self.car.distance_to_walls(self.walls)
        self.window.blit(self.background, (0, 0))
        self.car_group.update()
        self.track_group.draw(self.window)
        self.car_group.draw(self.window)
        self.finish_group.draw(self.window)
        distances = self.car.distance_to_walls(self.walls)
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
    MODE = "ai"
    all_replays = []
    env = Environment(MODE)
    scores= []

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    device = "cpu"

    print(device)
    
    if MODE == "ai":

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        for i in range(0, 10000):
            
            current_game = True
            BATCH_SIZE = 128
            GAMMA = 0.99
            EPS_START = 0.9
            EPS_END = 0.05
            EPS_DECAY = 1000
            TAU = 0.005
            LR = 1e-4
            
            # TODO: remove this and check for t in count()
            t = 0
            
            n_actions = 9

            # TODO: Can chose to let env.reset return state or grab it in another way.
            #state, info = env.reset()
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            n_observations = 3 #len(state[0])

            # TODO: Set device
            policy_net = DQN(n_observations, n_actions).to(device)
            target_net = DQN(n_observations, n_actions).to(device)
            target_net.load_state_dict(policy_net.state_dict())

            optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
            memory = ReplayMemory(10000)

            steps_done = 0

            episode_durations = []

            # Move this?
            def optimize_model():
                if len(memory) < BATCH_SIZE:
                    return
                
                Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
                
                transitions = memory.sample(BATCH_SIZE)
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                            if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net

                # TODO: change to one-hot encoding maybe?
                # TODO: check what happens in this function
                # TODO: Scale state_batch?????

                state_action_values = policy_net(state_batch).gather(1, action_batch)
                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1).values
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch

                # Compute Huber loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

            while current_game:

                # TODO: check inputs for select_action
                # We update the model in select actoin
                action = select_action(state, env, policy_net)
                print("actoin", action.item())
                # TODO: adept step to return certain items.
                # What should every type be, what do we need in terminated, truncated
                # Why do we return these variables instead of just checking them through the object
                # env.reward ?????
                observation, reward, terminated, truncated = env.step(action.item())
                
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                memory.push(state, action, next_state, reward)

                state = next_state
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
                target_net.load_state_dict(target_net_state_dict)
                if done:
                    print(" we get here")
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
        
            #all_replays.append(env.history)
            #replay_per_run = store_replay(all_replays)
            
            print("Finished")
            plot_durations(show_result=True)
            plt.ioff()
            plt.show()

    elif MODE == "player":
        current_game = True
        
        while current_game:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    current_game = False     
            keys = pygame.key.get_pressed()
            env.step(keys)

    elif MODE == "view":
        
        FILENAME = "replays/rpl0-20240717-201542.csv"
        coordinates, moves = read_replay(FILENAME)
        #TODO: use enumerate
        for i in range(0, len(moves)):
            env.step(moves[i])


