import pygame
from pygame.sprite import Sprite
import numpy as np

"""
# Putting in coordinates for checkpoints sucks
# In this function do the following:
# Load map
# Set points in pairs of two
# Draw lines between these points
# Save the points in this line
# Write to text file
"""

WINDOW_WIDTH    = 1920
WINDOW_HEIGHT   = 1080
WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

BLUE = (106, 159, 181)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)


def integer_points_on_line(p1, p2):
    """
    Return a list of integer points on the line between p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    points = []

    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))

    x_increment = dx / steps
    y_increment = dy / steps

    x = x1
    y = y1
    for _ in range(steps + 1):
        points.append((round(x), round(y)))
        x += x_increment
        y += y_increment

    points = list(set(points))
    points.sort()

    return points


class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

class Environment():
    """load and update the game, take in actions, keep score"""
    def __init__(self):
        self.width, self.height = 1920, 1080
        # TODO: put these under mode?
        clock = pygame.time.Clock()
        clock.tick_busy_loop(60)
        #self.walls = self._get_walls("track_1.csv")
        pygame.init()
        pygame.display.set_caption("Car sim :)")
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), WINDOW_SURFACE)
        self.background = pygame.image.load("assets/background2.png")
        self.track_group = self._load_obstacles()
        self.checkpoints = []
        
    def render(self) -> None:
        #TODO: this one seems extra
        self.window.blit(self.background, (0, 0))
        #pygame.draw.line(self.window, (255, 0, 0), (0, 300), (1800, 300))
        self.track_group.draw(self.window)
        for checkp in self.checkpoints:
            pygame.draw.line(self.window, (255, 0 ,0), checkp[0], checkp[1])
        pygame.display.flip()
        return None

    # What is returned?
    def _load_obstacles(self) -> None:
        self.track = Level("assets/track 3.png", WINDOW_WIDTH//2, WINDOW_HEIGHT//2)
        self.track_group = pygame.sprite.Group()
        self.track_group.add(self.track)
        return self.track_group

    # easy way to get the position of the track walls
    def _get_walls(self, track) -> np.array:
        return np.genfromtxt(track, delimiter=',', dtype = bool).reshape(1920, 1080)
    
    def add_checkpoints(self, checkpoint):
        self.checkpoints.append(checkpoint)

if __name__ == "__main__":

    env = Environment()

    current_game = True

    checkpoint = []

    while current_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                current_game = False     
            if event.type == pygame.MOUSEBUTTONDOWN:
                checkpoint.append(event.pos)
        
        if len(checkpoint) == 2:
            env.add_checkpoints(checkpoint)
        
        env.render()
        if len(checkpoint) == 2:
            checkpoint = []
    all_checkp = env.checkpoints

    with open(r"checkpoints\track 3.txt", "w") as fp:
        for point in all_checkp:
            fp.write("%s\n" % [point[0], point[1]])
           


