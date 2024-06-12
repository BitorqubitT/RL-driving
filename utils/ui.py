


def play_level(screen, sens1, sens2, sens3, speed, laps):
    lap_time = UIElement(
        center_position=(1840, 60),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text="Lap time",
    )

    sens1 = UIElement(
        center_position=(1840, 90),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"sens1: {sens1}",
    )

    sens2 = UIElement(
        center_position=(1840, 120),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"sens2: {sens2}",
    )

    sens3 = UIElement(
        center_position=(1840, 150),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"sens3: {sens3}",
    )

    speed = UIElement(
        center_position=(1840, 180),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"Speed: {speed}",
    )

    lap_counter = UIElement(
        center_position=(1840, 210),
        font_size=20,
        bg_rgb=BLUE,
        text_rgb=WHITE,
        text=f"Laps ({laps})",
    )

    lap_time.draw(screen)
    speed.draw(screen)
    lap_counter.draw(screen)
    sens1.draw(screen)
    sens2.draw(screen)
    sens3.draw(screen)

def create_surface_with_text(text, font_size, text_rgb, bg_rgb):
    """ Returns surface with text written on """
    font = pygame.freetype.SysFont("Courier", font_size, bold=True)
    surface, _ = font.render(text=text, fgcolor=text_rgb, bgcolor=bg_rgb)
    return surface.convert_alpha()

class UIElement(Sprite):
    """ An user interface element that can be added to a surface """

    def __init__(self, center_position, text, font_size, bg_rgb, text_rgb, action=None):
        """
        Args:
            center_position - tuple (x, y)
            text - string of text to write
            font_size - int
            bg_rgb (background colour) - tuple (r, g, b)
            text_rgb (text colour) - tuple (r, g, b)
            action - the gamestate change associated with this button
        """
        self.image = create_surface_with_text(text=text, font_size=font_size, text_rgb=text_rgb, bg_rgb=bg_rgb)
        self.rect = self.image.get_rect(center=center_position)
        self.action = action

    def draw(self, surface):
        surface.blit(self.image, self.rect)