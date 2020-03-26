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
