import pygame as py

class Font:
    """
    Font class returns a surface with rectangle object in
    a list for the given text. The size of the text is
    initialised in the Font object instantiation
    """

    def __init__(self, size):
        self.font = py.font.Font('freesansbold.ttf', size)

    def text_object(self, text):
        surface = self.font.render(text, True, (255, 255, 255))
        return [surface, surface.get_rect()]
