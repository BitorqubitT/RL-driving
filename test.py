import math
import sys
import pygame as pg

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pg.init()
beam_surface = pg.Surface((500, 500), pg.SRCALPHA)


def draw_beam(surface, angle, pos):
    c = math.cos(math.radians(angle))
    s = math.sin(math.radians(angle))

    flip_x = c < 0
    flip_y = s < 0
    filpped_mask = flipped_masks[flip_x][flip_y]
    
    # compute beam final point
    x_dest = 250 + 500 * abs(c)
    y_dest = 250 + 500 * abs(s)

    beam_surface.fill((0, 0, 0, 0))

    # draw a single beam to the beam surface based on computed final point
    pg.draw.line(beam_surface, BLUE, (250, 250), (x_dest, y_dest))
    beam_mask = pg.mask.from_surface(beam_surface)

    # find overlap between "global mask" and current beam mask
    offset_x = 250 - pos[0] if flip_x else pos[0] - 250
    offset_y = 250 - pos[1] if flip_y else pos[1] - 250
    hit = filpped_mask.overlap(beam_mask, (offset_x, offset_y))
    if hit is not None and (hit[0] != pos[0] or hit[1] != pos[1]):
        hx = 499 - hit[0] if flip_x else hit[0]
        hy = 499 - hit[1] if flip_y else hit[1]
        hit_pos = (hx, hy)

        pg.draw.line(surface, BLUE, pos, hit_pos)
        pg.draw.circle(surface, GREEN, hit_pos, 3)
        #pg.draw.circle(surface, (255, 255, 0), mouse_pos, 3)


surface = pg.display.set_mode((500, 500))
#mask_surface = pg.image.load("../assets/mask.png")
mask_surface = pg.Surface((500, 500), pg.SRCALPHA)
mask_surface.fill((255, 0, 0))
pg.draw.circle(mask_surface, (0, 0, 0, 0), (250, 250), 100)
pg.draw.rect(mask_surface, (0, 0, 0, 0), (170, 170, 160, 160))

mask = pg.mask.from_surface(mask_surface)
mask_fx = pg.mask.from_surface(pg.transform.flip(mask_surface, True, False))
mask_fy = pg.mask.from_surface(pg.transform.flip(mask_surface, False, True))
mask_fx_fy = pg.mask.from_surface(pg.transform.flip(mask_surface, True, True))
flipped_masks = [[mask, mask_fy], [mask_fx, mask_fx_fy]]

clock = pg.time.Clock()

while True:
    for e in pg.event.get():
        if e.type == pg.QUIT:
            pg.quit()
            sys.exit()

    mouse_pos = pg.mouse.get_pos()

    surface.fill((0, 0, 0))
    surface.blit(mask_surface, mask_surface.get_rect())

    for angle in range(0, 359, 30):
        draw_beam(surface, angle, mouse_pos)

    pg.display.update()
    clock.tick(30)