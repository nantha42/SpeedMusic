import pygame as py
import pygame.gfxdraw
from instrument import Instrument


class InstrumentPanel:
    """Instrument Panel displays the types of instruments used
        in the project and also ables to add more instruments to
        the file."""

    def __init__(self):
        self.width = 160
        self.height = 500
        py.font.init()
        self.surface_panel = py.Surface((self.width, self.height))
        self.state = {"isopen": False,"delete":False}
        self.instruments_used = [0]
        self.instruments_sprites = []
        self.requests = []
        self.surface_panel_images = {"add": py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/add_m.png"),
                                     "delete": py.image.load(
                                         "/Users/nantha/Projc/my_projects/FastMusic/images/delete.png")}
        # responses contains dictionary object
        # for example: {"program":43}
        self.responses = []
        self.seekresponse = False
        self.build_instruments()
        self.predraw()

    def build_instruments(self):
        """Reads the indices of instruments used and
            fills the instruments_sprites with sprites"""
        all_instruments = self.read_instrumentslist()
        for i in range(len(self.instruments_used)):
            self.instruments_sprites.append(Instrument(self.instruments_used[i],
                                                       i,
                                                       all_instruments[self.instruments_used[i]]))
            # print(all_instruments[i], " Builded")

    def add_instrument(self, programno):
        all_instruments = self.read_instrumentslist()
        programname = all_instruments[programno]
        # print("Read from all_instruments", programname)

        self.instruments_used.append(programno)
        self.instruments_sprites.append(Instrument(programno,
                                                   len(self.instruments_used) - 1,
                                                   programname
                                                   ))

    def predraw(self):
        background = (30,0,40)
        if self.state["delete"] :
            background = ((200,0,0))
        self.surface_panel.fill((30, 30, 40))
        py.draw.line(self.surface_panel, (255, 255, 255), (0, 0), (0, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (0, self.height), (self.width, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, self.height), (self.width, 0), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, 0), (0, 0), 3)  # down
        self.surface_panel.blit(self.surface_panel_images["add"], (8, 8))
        self.surface_panel.blit(self.surface_panel_images["delete"], (32, 8))

        for i in range(len(self.instruments_sprites)):
            self.instruments_sprites[i].rect.topleft = (10, 30 + i * 50)
            self.instruments_sprites[i].predraw(background)
            self.surface_panel.blit(self.instruments_sprites[i].image, self.instruments_sprites[i].rect)

    def draw(self):
        # print("Drawn state delete:",self.state["delete"])
        self.predraw()

    def read_instrumentslist(self):
        """Read the instrumentlist and returns the list of
        available instruments"""
        available_instruments = []
        with open("instrumentslist.txt", "r") as file:
            for line in file:
                splited = line.split(" ")
                name = ""
                for j in splited[1:]:
                    name += j
                available_instruments.append(name)
        return available_instruments

    def handle_events(self, pos):
        """Handles the mouse click made on the
        instrument panel surface"""
        x, y = pos
        # print(x, y)

        # checks if any instrument is clicked
        instrument_clicked = False
        for sprite in self.instruments_sprites:
            if sprite.rect.collidepoint((x, y)):
                if not self.state["delete"]:
                    self.requests.append({"clicked": sprite.index})
                else:
                    self.requests.append({"delete": sprite.index})
                    self.state["delete"] = False
                instrument_clicked = True
                break
        if self.state["delete"]:
            if not instrument_clicked:
                # print("outclick")
                self.state["delete"] = False
                self.predraw()

        if x > 8 and x < 8 + 16 and y > 8 and y < 8 + 16:
            self.requests.append("instruments")
            self.seekresponse = True
            pass
        elif x > 32 and x < 32 + 16 and y > 8 and y < 8 + 16:
            self.state["delete"] = True
            pass
        # delete box
        # print("Handled by Instrument panel")

    def __handle_response(self, response):
        if 'program' in response.keys():
            programno = response['program']
            self.add_instrument(programno)
            if len(self.responses) == 0:
                self.seekresponse = False
            pass

    def update(self):
        if self.seekresponse == True:
            if len(self.responses) > 0:
                for response in self.responses:
                    self.__handle_response(response)
                    pass
        self.draw()
