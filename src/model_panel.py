import os
import json
import pygame.sprite
from pygame import Surface
from font import *
import pygame as py


class ModelPanel(pygame.sprite.Sprite):
    def __init__(self,):
        pygame.sprite.Sprite.__init__(self)
        self.models_available = []
        self.requests = []
        self.selected_model_index = 0
        self.width = 150
        self.height = 30
        self.image = Surface((150, 30))
        self.opened = False
        self.rect = self.image.get_rect()
        self.__initiate()

    def __initiate(self):
        jfile = open("./models/info.json")
        f = json.load(jfile)

        for obj in f:
            self.models_available.append([[x,y] for x,y in obj.items()][0])

        rect = py.Rect((0,0,0,0))
        rect.x = 0
        rect.y = 0
        rect.width = self.width
        rect.height = self.height
        self.image.fill((30,30,40))
        py.draw.rect(self.image, (255, 255, 255), rect,1)

        font = Font(14)
        surf,rec = font.text_object(self.models_available[self.selected_model_index][0])
        self.image.blit(surf,(5,5))

    def handle_events(self, pos):
        print("Event Handled")
        if self.opened == False:
            self.requests.append("open")
        else:
            self.requests.append("close")

    def update(self):
        font = Font(14)
        self.image.fill((30, 30, 40))
        surf, rec = font.text_object(self.models_available[self.selected_model_index][0])
        self.image.blit(surf, (5, 5))

    def get_models_list(self):
        return self.models_available

    def get_model_detail(self):
        print(self.models_available[self.selected_model_index])
        return self.models_available[self.selected_model_index]





