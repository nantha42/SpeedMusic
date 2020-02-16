import pygame as py
from font import Font


class Instrument(py.sprite.Sprite):

    def __init__(self, program, i,iname):
        py.sprite.Sprite.__init__(self)
        self.image = py.Surface([140, 40])
        self.rect = self.image.get_rect()
        self.index = i
        self.program = program
        self.iname = iname
        self.namefont = Font(13)
        self.predraw()

    def predraw(self):
        self.image.fill((30,0,40))
        py.draw.line(self.image, (255, 255, 255), (0, 0), (0, 40),3)  # down
        py.draw.line(self.image, (255, 255, 255), (0, 40), (140, 40),3)  # down
        py.draw.line(self.image, (255, 255, 255), (140, 40), (140, 0),3)  # down
        py.draw.line(self.image, (255, 255, 255), (140, 0), (0, 0),3)  # down
        surf, rect = self.namefont.text_object(str(self.index)+ self.iname)
        rect.topleft=(10,10)
        self.image.blit(surf, rect)

