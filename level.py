import pygame
import pygame.freetype
from pygame.sprite import Sprite

"""
    This contains the game environment for a simple car racing game.
    The graphics are seperated from the game logic.
    TODO: Check how to write good intro, name the different classes?
"""

#TODO: put in other file
class Level(Sprite):
    def __init__(self, image: str, x: int, y: int):
        Sprite.__init__(self)
        self.image = pygame.image.load(image).convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y) 