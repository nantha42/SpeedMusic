import pygame as py
import pygame.gfxdraw
from instrument import Instrument
from font import Font
from add_instruments_panel import  NameBox

class ChangeModelPanel:

    def __init__(self,models_list):

        self.height = 500
        self.width = 600
        self.models_list = models_list
        self.surface_panel = py.Surface((self.width, self.height))
        self.rect = self.surface_panel.get_rect()
        self.itype_height = self.height - 20
        self.itype_width = self.width / 2 - 50
        self.requests = []
        self.surface_itype = py.Surface((self.itype_width, self.itype_height))
        self.spritebox_models = []
        self.result = []

        #self.surface_typeitems = py.Surface((self.typeitems_width, self.typeitems_height))
        for i in range(len(self.models_list)):
            temp = NameBox(self.models_list[i][0], i)
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
            self.surface_panel.blit(box.image,box.rect)


    def handle_events(self,pos):
        x,y = pos
        selected_some = False
        id = -1
        for sprite in self.spritebox_models:
            if sprite.rect.collidepoint(x, y):
                id = sprite.id
                selected_some = True
                break

        if selected_some :
            self.requests.append("close")
            self.result = [id]

        print("Change_Model_panel",x,y,selected_some)



