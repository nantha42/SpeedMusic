import pygame as py
import pygame.gfxdraw
from instrument import Instrument
from font import Font
from add_instruments_panel import NameBox
import os


class FilePanelListener:

    def __init__(self):
        self.cwd = "../midis/"
        self.files_list = os.listdir(self.cwd)
        self.requests = []
        self.result = []
        self.opened = False

    def handle_events(self, pos):
        # print("Event Handled")
        if self.opened == False:
            self.requests.append("open")
        else:
            self.requests.append("close")

class FilePanelWindow:
    def __init__(self):
        self.height = 500
        self.width = 600
        self.file_list = os.listdir("documents/")
        self.surface_panel = py.Surface((self.width, self.height))
        self.rect = self.surface_panel.get_rect()
        self.requests = []
        self.result = []
        self.spritebox_models = []
        if len(self.file_list) == 0:
            self.requests.append("close")
            self.result.append(-1)

        for i in range(len(self.file_list)):
            temp = NameBox(self.file_list[i], i)
            temp.rect.topleft = (10, i * 30 + 4)
            self.spritebox_models.append(temp)
        self.__predraw()

    def __predraw(self):
        self.surface_panel.fill((30, 30, 40))
        py.draw.line(self.surface_panel, (255, 255, 255), (0, 0), (0, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (0, self.height), (self.width, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, self.height), (self.width, 0), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, 0), (0, 0), 3)  # down

    def draw(self):
        self.__predraw()
        for box in self.spritebox_models:
            self.surface_panel.blit(box.image, box.rect)

    def handle_events(self, pos):
        x, y = pos
        selected_some = False
        id = -1
        for sprite in self.spritebox_models:
            if sprite.rect.collidepoint(x, y):
                id = sprite.id
                selected_some = True
                break

        if selected_some:
            self.requests.append("close")
            self.result = [id]
