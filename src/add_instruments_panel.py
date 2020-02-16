import pygame as py
import pygame.gfxdraw
from instrument import Instrument
from font import Font


class NameBox(py.sprite.Sprite):

    def __init__(self, name, id):
        py.sprite.Sprite.__init__(self)
        self.image = py.Surface((140, 20))
        self.rect = self.image.get_rect()
        print(type(self.rect))
        self.name = name
        self.id = id
        self.__predraw()
        pass

    def __predraw(self):
        self.image.fill((30, 30, 40))
        fonter = Font(12)
        img, rt = fonter.text_object(self.name)
        rt.topleft = (10, 10)
        self.image.blit(img, rt)

    def select(self):
        self.image.fill((30, 35, 50))
        fonter = Font(12)
        img, rt = fonter.text_object(self.name)
        rt.topleft = (10, 10)
        self.image.blit(img, rt)

    def deselect(self):
        self.__predraw()


class AddInstrumentPanel:
    """Instrument Panel displays the types of instruments used
        in the project and also ables to add more instruments to
        the file."""

    def __init__(self):
        self.height = 500
        self.width = 600

        self.surface_panel = py.Surface((self.width, self.height))
        self.rect = self.surface_panel.get_rect()
        self.itype_height = self.height - 20
        self.itype_width = self.width / 2 - 50

        self.surface_itype = py.Surface((self.itype_width, self.itype_height))

        self.typeitems_height = self.height - 20
        self.typeitems_width = self.width / 2 - 50

        self.surface_typeitems = py.Surface((self.typeitems_width, self.typeitems_height))
        self.instrument_types = [
            "Piano",
            "Chromatic Percussion",
            "Organ",
            "Guitar",
            "Bass",
            "String",
            "Ensemble",
            "Brass",
            "Reed",
            "Pipe",
            "SynthLead",
            "SynthPad",
            "SynthEffect",
            "Ethnic",
            "Percussive",
            "SoundEffects"
        ]
        self.requests = []
        self.spritebox_itypes = []
        self.spritebox_itypeitems = []
        self.instruments_for_itypes = []
        self.selected_itype = -1
        self.result = None
        for i in range(len(self.instrument_types)):
            temp = NameBox(self.instrument_types[i], i)
            temp.rect.topleft = (10, i * 30 + 4)
            self.spritebox_itypes.append(temp)
        self.__predraw()
        pass

    def __predraw(self):
        py.draw.line(self.surface_panel, (255, 255, 255), (0, 0), (0, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (0, self.height), (self.width, self.height), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, self.height), (self.width, 0), 3)  # down
        py.draw.line(self.surface_panel, (255, 255, 255), (self.width, 0), (0, 0), 3)  # down

        py.draw.line(self.surface_itype, (255, 255, 255), (0, 0), (0, self.itype_height), 3)  # down
        py.draw.line(self.surface_itype, (255, 255, 255), (0, self.itype_height), (self.itype_width, self.itype_height),
                     3)  # down
        py.draw.line(self.surface_itype, (255, 255, 255), (self.itype_width, self.itype_height), (self.itype_width, 0),
                     3)  # down
        py.draw.line(self.surface_itype, (255, 255, 255), (self.itype_width, 0), (0, 0), 3)  # down

        py.draw.line(self.surface_typeitems, (255, 255, 255), (0, 0), (0, self.typeitems_height), 3)  # down
        py.draw.line(self.surface_typeitems, (255, 255, 255), (0, self.typeitems_height),
                     (self.typeitems_width, self.typeitems_height), 3)  # down
        py.draw.line(self.surface_typeitems, (255, 255, 255), (self.typeitems_width, self.typeitems_height),
                     (self.typeitems_width, 0), 3)  # down
        py.draw.line(self.surface_typeitems, (255, 255, 255), (self.typeitems_width, 0), (0, 0), 3)  # down

    def __read_instrumentslist(self):
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
        """Gets mouse click position and if any instrument item is clicked
        then place a request to close and stores the instrument id in self.result"""
        x, y = pos
        if x > 20 and x < 20 + self.itype_width and y > 20 and y < 20 + self.itype_height:
            x = x - 20
            y = y - 20
            id = -1
            selected_some = False
            for sprite in self.spritebox_itypes:
                if sprite.rect.collidepoint(x, y):
                    # print("Clicked",sprite.name)
                    id = sprite.id
                    selected_some = True
                    break
            if selected_some:
                if self.selected_itype != -1:
                    self.spritebox_itypes[self.selected_itype].deselect()
                self.spritebox_itypes[id].select()
                self.selected_itype = id

            if id != -1:
                ins = self.__read_instrumentslist()
                self.instruments_for_itypes = ins[id * 8:id * 8 + 8]
                print(self.instruments_for_itypes)
                self.spritebox_itypeitems = []
                print("Length", len(self.instruments_for_itypes))
                for i in range(len(self.instruments_for_itypes)):
                    temp = NameBox(self.instruments_for_itypes[i][:-1], i)
                    temp.rect.topleft = (10, i * 30 + 4)
                    self.spritebox_itypeitems.append(temp)

                print("Spritebox_itypeitems", self.spritebox_itypeitems)

        elif x > 270 and x < 270 + self.typeitems_width and y > 20 and y < 20 + self.typeitems_height:
            x = x - 270
            y = y - 20
            id = -1
            for sprite in self.spritebox_itypeitems:
                if sprite.rect.collidepoint((x,y)):
                    id = sprite.id
                    break
            self.result = self.selected_itype*8+id
            self.requests.append("close")

    def __get_instrument_items(self, i):
        """returns items belonging particular instrument type"""
        all_instruments = self.__read_instrumentslist()
        return [x for x in all_instruments[i * 8:i * 8 + 8]]

    def draw(self):
        self.surface_panel.fill((30, 30, 40))
        self.surface_itype.fill((30, 30, 40))
        self.surface_typeitems.fill((30, 30, 40))
        self.__predraw()

        for i in range(len(self.spritebox_itypes)):
            self.surface_itype.blit(self.spritebox_itypes[i].image, self.spritebox_itypes[i].rect)

        for i in range(len(self.spritebox_itypeitems)):
            self.surface_typeitems.blit(self.spritebox_itypeitems[i].image, self.spritebox_itypeitems[i].rect)

        self.surface_panel.blit(self.surface_itype, (20, 20))
        self.surface_panel.blit(self.surface_typeitems, (270, 20))
        pass
