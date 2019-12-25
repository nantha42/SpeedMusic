import pygame as py
import pygame.gfxdraw
import time

class Pianoroll:
    def __init__(self):
        self.height = 500
        self.width = 800
        self.pianoroll = py.Surface((self.width, self.height))
        self.controls = {"h_zoom": 1}
        py.font.init()
        self.timeFont = Font(10)
        self.draw_grids()




    def draw_grids(self):
        roll_index = 0
        key_color = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        key_color.reverse()
        self.pianoroll.fill((30, 30, 40))

        for j in range(10, self.height, 10):
            #draw horizontal lines
            py.gfxdraw.line(self.pianoroll, 20, j, self.width-20, j, (255, 255, 255, 30))

            #below code is for rendering the piano keys on the left
            if (key_color[roll_index] == 0):
                py.draw.rect(self.pianoroll, (255, 255, 255), (0, j + 2, 15, 8))
                py.draw.rect(self.pianoroll,(30,30,50),(20,j+2,self.width-20,8))
            else:
                py.draw.rect(self.pianoroll, (0, 0, 0), (0, j + 2, 15, 8))
            roll_index = (roll_index + 1) % 12
        index = 0
        for i in range(20, self.width, int(20*self.controls["h_zoom"])):
            #draw vertical lines
            py.gfxdraw.line(self.pianoroll, i, 10, i, self.height-10, (255, 255, 255, 30))

        for i in range(0,self.width,int(20*self.controls["h_zoom"]*32)):
            #creates color differentiation for the black key rows and white key rows
            surf,rect = self.timeFont.text_object(str(index))
            rect.topleft = (i+20,0)
            self.pianoroll.blit(surf,rect)
            index +=1

class Font:

    def __init__(self,size):
        self.font = py.font.Font('freesansbold.ttf',size)

    def text_object(self,text):
        surface = self.font.render(text,True,(255,255,255))
        return [surface,surface.get_rect()]


class Display:

    def __init__(self):
        py.display.init()

        self.width = 1000
        self.height = 800
        self.win = py.display.set_mode((self.width,self.height))
        self.pianoroll = Pianoroll()
        self.roll_index = 0
        self.j = 10
        self.controls = {"play":False}
        self.event_handled = False
        self.quit = False

    def draw(self):
        self.win.blit(self.pianoroll.pianoroll, (180, 100))
        self.draw_controls()


    def draw_controls(self):
        button_play = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/play.png")
        button_pause = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/pause.png")
        if self.controls["play"]==False:
            self.win.blit(button_play,(500,30))
        else:
            self.win.blit(button_pause,(500,30))

    def event_handler(self):

        for event in py.event.get():
            if event.type == py.QUIT:
                self.quit = True


            if event.type == py.KEYDOWN:
                self.event_handled = True
                if event.key == py.K_UP:
                    if self.pianoroll.controls["h_zoom"] >= 1:
                        self.pianoroll.controls["h_zoom"] += 1
                        self.pianoroll.draw_grids()
                    else:
                        self.pianoroll.controls["h_zoom"] *= 2
                        self.pianoroll.draw_grids()

                if event.key == py.K_DOWN:
                    if self.pianoroll.controls["h_zoom"] > 1:
                        self.pianoroll.controls["h_zoom"] -= 1
                        self.pianoroll.draw_grids()

                    elif (self.pianoroll.controls["h_zoom"] > 0.0625):
                        self.pianoroll.controls["h_zoom"] /= 2
                        self.pianoroll.draw_grids()
                        print(self.pianoroll.controls["h_zoom"])

            if event.type == py.MOUSEBUTTONUP:
                x, y = py.mouse.get_pos()
                print(x, y)
                if x > 500 and x < 532 and y > 30 and y < 62:
                    self.controls["play"] = not self.controls["play"]

    def run(self):
        quit = False

        while(not self.quit):
            self.win.fill((30,30,40))
            self.draw()
            self.event_handler()

            if self.event_handled == True:
                py.display.update()
                self.event_handled = False


if __name__ == '__main__':
    d = Display()
    d.run()
