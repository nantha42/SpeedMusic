import pygame as py
import sys
from pianoroll import *
from pypianoroll import *
from jarvis import *
from instrument_panel import *
import time
from add_instruments_panel import *


class Display:

    def __init__(self, mes, file=""):
        py.display.init()
        self.width = 1000
        self.height = 800
        self.win = py.display.set_mode((self.width, self.height))
        self.measure_limit = mes
        self.measure_length = 91
        self.pianoroll = Pianoroll(self.measure_limit)
        self.roll_index = 0
        self.j = 10
        self.controls = {"right_click": False, "play": False, "move_piano_roll_left": False, "update_ipanel": True,
                         "move_piano_roll_right": False, "playing": False, "show_map": False}
        self.events = {"update_pianoroll": True}
        self.event_handled = True
        self.quit = False
        self.window = None
        self.instrument_panel = InstrumentPanel()

        self.openfile = False
        if file == "":
            self.openfile = False
            self.file = ""

        else:
            self.openfile = True
            self.file = file
            try:
                parsed_track = self.parse_first_track(file)
                print("ParsedTrack", parsed_track)
                self.pianoroll.load_file(parsed_track, self.measure_limit * self.measure_length)
            except Exception:
                print("No file in that name")
                pass

        print(self.file)
        py.mixer.init()
        self.button_play = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/play.png")
        self.button_pause = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/pause.png")
        self.button_note = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/" + str(
            2 ** self.pianoroll.controls["nthnote"]) + "_on.png")
        self.button_addm = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/add_m.png")
        self.button_subm = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/sub_m.png")
        self.symbol_measure = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/measure.png")

    def draw(self):
        # For minimizing the frequency of the pianoroll surface updates

        if self.events["update_pianoroll"] or self.pianoroll.controls["bar_move_velocity"] != 0:
            self.win.fill((30, 30, 40))
            self.pianoroll.update()
            self.events["update_pianoroll"] = False
            self.win.blit(self.pianoroll.pianoroll, (180, 100))
            self.draw_controls()
            self.win.blit(self.instrument_panel.surface_panel, (10, 100))
            if self.window != None:
                self.window.draw()
                self.window.rect.topleft = (100, 100)
                self.win.blit(self.window.surface_panel, self.window.rect)
            py.display.update()
        if self.controls["update_ipanel"]:
            self.controls["update_ipanel"] = False
            self.instrument_panel.draw()

    def draw_controls(self):
        self.button_note = py.image.load("/Users/nantha/Projc/my_projects/FastMusic/images/" + str(
            2 ** self.pianoroll.controls["nthnote"]) + "_on.png")

        if self.controls["play"] == False:
            self.win.blit(self.button_play, (500, 30))
        else:
            self.win.blit(self.button_pause, (500, 30))

        self.win.blit(self.button_note, (450, 30))
        self.win.blit(self.symbol_measure, (550, 30))
        self.win.blit(self.button_addm, (580, 30))
        self.win.blit(self.button_subm, (580, 46))

        if self.controls["move_piano_roll_left"] or self.controls["move_piano_roll_right"]:
            self.pianoroll.controls["make_velocity_0"] = False
            if self.controls["move_piano_roll_left"]:
                if self.pianoroll.controls["fast_move_piano_roll"]:
                    if self.pianoroll.controls["bar_move_velocity"] > -30.0:
                        self.pianoroll.controls["bar_move_velocity"] -= 1.0
                else:
                    if self.pianoroll.controls["bar_move_velocity"] > -10.0:
                        self.pianoroll.controls["bar_move_velocity"] -= 0.2
                    else:
                        self.pianoroll.controls["bar_move_velocity"] += 1.0

            if self.controls["move_piano_roll_right"]:
                if self.pianoroll.controls["fast_move_piano_roll"]:
                    if self.pianoroll.controls["bar_move_velocity"] < 30.0:
                        self.pianoroll.controls["bar_move_velocity"] += 1.0
                else:
                    if self.pianoroll.controls["bar_move_velocity"] < 10.0:
                        self.pianoroll.controls["bar_move_velocity"] += 0.2
                    else:
                        self.pianoroll.controls["bar_move_velocity"] -= 1.0
            # print("B",self.pianoroll.controls["bar_move_velocity"])
        else:
            self.pianoroll.controls["make_velocity_0"] = True

    def event_handler(self):

        for event in py.event.get():
            if event.type == py.QUIT:
                self.quit = True

            if event.type == py.MOUSEMOTION:
                if self.controls["right_click"]:
                    x, y = py.mouse.get_pos()
                    if x >= 200 and x <= 980 and y >= 110 and y <= 590:
                        s = int((x - 200))
                        k = int((y - 110))
                        s = int((s - self.pianoroll.controls["timebar"]) / (20 * self.pianoroll.controls["h_zoom"]))
                        k = int(k / 10)
                        self.pianoroll.deletenote(s, k)
                        self.events["update_pianoroll"] = True

            if event.type == py.MOUSEBUTTONDOWN:
                if event.button == 3:
                    self.controls["right_click"] = True

            if event.type == py.KEYDOWN:

                self.event_handled = True
                if event.key == py.K_1:
                    self.pianoroll.controls["nthnote"] = 0
                if event.key == py.K_2:
                    self.pianoroll.controls["nthnote"] = 1
                if event.key == py.K_3:
                    self.pianoroll.controls["nthnote"] = 2
                if event.key == py.K_4:
                    self.pianoroll.controls["nthnote"] = 3
                if event.key == py.K_5:
                    self.pianoroll.controls["nthnote"] = 4
                if event.key == py.K_6:
                    self.pianoroll.controls["nthnote"] = 5
                if event.key == py.K_p:
                    self.pianoroll.play_notes()
                if event.key == py.K_o:
                    try:
                        self.load_from_npy("generated.npy")
                        self.events["update_pianoroll"] = True
                    except Exception:
                        pass

                if event.key == py.K_c:
                    self.pianoroll.clear_current_track()
                    self.events["update_pianoroll"] = True


                if event.key == py.K_g:
                    np.save("temp.npy", self.pianoroll.notes)
                    print("Saved Temp")
                    jarv = Melody("deep_flight_model.h5")
                    print(jarv.model.summary())
                    jarv.input_length = 32
                    # jarv.use_model()
                    jarv.generate_tune()

                if event.key == py.K_f:
                    np.save("temp.npy", self.pianoroll.notes)
                    print("Saved Temp")
                    jarv = Melody("newmodel.h5")
                    print(jarv.model.summary())
                    jarv.input_length = 128
                    jarv.use_model()

                if event.key == py.K_e:
                    """Uses only the notes that is present in the 
                    pianoroll.notes_index"""
                    jarv = Melody("256model.h5")
                    use_notes = list()

                    for note_index in self.pianoroll.notes_index[self.pianoroll.selected_track]:
                        s, t = note_index
                        # print(note_index, note_index[0] * 6 + self.pianoroll.notes[self.pianoroll.selected_track][s][t])
                        use_notes.append(note_index[0] * 6 + self.pianoroll.notes[self.pianoroll.selected_track][s][t])
                    jarv.input_length = 128
                    jarv.input_data_to_model(self.pianoroll.notes)
                    jarv.random_noise()

                if event.key == py.K_m:
                    self.controls["show_map"] = not self.controls["show_map"]

                if event.key == py.K_t:
                    parsed_track = self.parse_first_track(self.file)
                    print("ParsedTrack", parsed_track)
                    self.pianoroll.load_file(parsed_track, self.measure_limit * self.measure_length)

                if event.key == py.K_r:
                    for i in range(self.pianoroll.notes.shape[1]):
                        ter = self.pianoroll.notes[:, i]
                        for j in range(len(ter)):
                            if ter[j] != -1:
                                ter[j + 1:] = -1
                    self.events["update_pianoroll"] = True

                if event.key == py.K_q:
                    np.save('train_data.npy', self.pianoroll.notes)
                    print("Saved Successfully")

                if event.key == py.K_s:
                    np.save("saved_advance/" + str(int(time.time())), self.pianoroll.notes)
                    print("Saved successfully")

                if event.key == py.K_UP:
                    if self.pianoroll.controls["h_zoom"] >= 1:
                        self.pianoroll.controls["h_zoom"] += 1
                        self.events["update_pianoroll"] = True
                        print(self.pianoroll.controls["h_zoom"])

                    else:
                        self.pianoroll.controls["h_zoom"] *= 2
                        self.events["update_pianoroll"] = True
                    # print(self.pianoroll.controls["h_zoom"])

                if event.key == py.K_DOWN:

                    if self.pianoroll.controls["h_zoom"] > 1:
                        self.pianoroll.controls["h_zoom"] -= 1
                        self.events["update_pianoroll"] = True

                    elif self.pianoroll.controls["h_zoom"] > 0.25:
                        self.pianoroll.controls["h_zoom"] /= 2
                        self.events["update_pianoroll"] = True
                    # print(self.pianoroll.controls["h_zoom"])

                if event.key == py.K_LEFT:
                    self.events["update_pianoroll"] = True
                    self.controls["move_piano_roll_left"] = True

                if event.key == py.K_RIGHT:
                    self.events["update_pianoroll"] = True
                    self.controls["move_piano_roll_right"] = True

                if event.key == py.K_RSHIFT:
                    print("shift pressed")
                    self.pianoroll.controls["fast_move_piano_roll"] = True

                if event.key == py.K_SPACE:
                    self.controls["play"] = not self.controls["play"]

            if event.type == py.KEYUP:
                self.events["update_pianoroll"] = True
                if event.key == py.K_LEFT:
                    self.controls["move_piano_roll_left"] = False

                if event.key == py.K_RIGHT:
                    self.controls["move_piano_roll_right"] = False

                if event.key == py.K_RSHIFT:
                    self.pianoroll.controls["fast_move_piano_roll"] = False

            if event.type == py.USEREVENT:
                self.controls["play"] = False
                self.controls["playing"] = False
                py.mixer.music.stop()

            if event.type == py.MOUSEBUTTONUP:
                self.event_handled = True
                self.events["update_pianoroll"] = True
                x, y = py.mouse.get_pos()
                # print(x, y)
                # On cliking the play button
                if self.window != None:
                    if x > 100 and x < 100 + self.window.width and y > 100 and y < 100 + self.window.height:
                        self.window.handle_events((x - 100, y - 100))

                if x > 10 and x < 175 and y > 100 and y < 600:
                    self.instrument_panel.handle_events((x - 10, y - 100))

                if x > 500 and x < 532 and y > 30 and y < 62:
                    self.controls["play"] = not self.controls["play"]

                # On clicking the pianoroll for entering notes
                if x >= 200 and x <= 980 and y >= 110 and y <= 590 and self.controls["right_click"] == False:
                    s = int((x - 200))
                    k = int((y - 110))
                    s = int((s - self.pianoroll.controls["timebar"]) / (20 * self.pianoroll.controls["h_zoom"]))
                    k = int(k / 10)
                    self.pianoroll.enternote(s, k)

                if x >= 580 and x <= 596 and y >= 30 and y <= 62:
                    if y >= 30 and y <= 46:
                        self.pianoroll.measures += 1
                        self.pianoroll.send_event("measure_increased")
                        # print("measure_increased")
                    elif y >= 46 and y <= 62:
                        if self.pianoroll.measures > 1:
                            self.pianoroll.measures -= 1
                            self.pianoroll.send_event("measure_decreased")
                        # print("measure_decreased")
                if event.button == 3:
                    self.controls["right_click"] = False
                    # print(self.pianoroll.notes)

    def load_from_npy(self, file):
        array = np.load(file)
        self.pianoroll.notes = []
        self.pianoroll.notes.append(array)
        self.pianoroll.notes_index = []
        for notes in self.pianoroll.notes:
            note_index = []
            for i in range(notes.shape[1]):
                row = notes[:, i]
                for v in range(len(row)):
                    if row[v] != -1:
                        note_index.append([v, i])
                        break
            self.pianoroll.notes_index.append(note_index)

    def play(self):
        if self.controls["play"] == True and self.controls["playing"] == False:
            self.pianoroll.play_notes(self.controls["show_map"])
            py.mixer.music.load("create1.mid")
            py.mixer.music.set_endevent(py.USEREVENT)
            self.controls["playing"] = True
            py.mixer.music.play()
            # print("index", self.pianoroll.notes_index)

        elif self.controls["play"] == False:
            self.controls["playing"] = False
            py.mixer.music.stop()

    def find_closer(self, x):
        """
        Returns the closest resemblance of the note type
        such as whole(96),half(48),half+quarter.
        Currently ex_dura feature is turned off, because of
        reducing complexity
        :param x:
        :return:
        """
        # ex_dura = [96,72,48,36,24,18,12,9,6,4,3]
        dura = [self.measure_length, self.measure_length / 2, self.measure_length / 4, self.measure_length / 8,
                self.measure_length / 16, self.measure_length / 32]
        dura.reverse()
        dura = np.array(dura)
        my_dura = [0.125, 0.25, 0.5, 1, 2, 4]
        my_dura = np.array(my_dura)
        duration = my_dura[np.argmin(np.abs(x - dura))] * 8
        # print(duration)
        return duration

    def get_notes(self, key_row):
        """
        Returns a list of list of starting position
        and duration of a particular note in the entire track
        :param key_row:
        :return:
        """
        nonzeros = np.nonzero(key_row)[0]
        # print("Nonzeros1",nonzeros)
        notes = []
        if len(nonzeros) > 0:
            start = int(nonzeros[0])
            prestart = start
            length = 0
            # tuple stores length start,length
            for i in range(1, len(nonzeros)):
                if start + 1 == nonzeros[i] and i < len(nonzeros) - 1:
                    length += 1
                    start += 1
                else:
                    notes.append([prestart, self.find_closer(length)])
                    length = 0
                    prestart = int(nonzeros[i])
                    start = prestart
                pass
            print("Notes", notes)
            return notes
        else:
            return []
        # print(nonzeros.shape)

    def parse_first_track(self, file):
        """
        Parses the midi file and converts it into the format
        required for my pianoroll function.
        :param file:
        :return:
        """
        track = Multitrack(file).tracks[0]
        print("Track", track.pianoroll)
        shortened_track = []
        for i in range(128):
            range_array = self.get_notes(track.pianoroll[:self.measure_limit * self.measure_length, i])
            # print(range_array)
            if len(range_array) > 0:
                shortened_track.append([i, range_array])
        shortened_track = np.array(shortened_track)
        # print("shape", shortened_track.shape)
        modified_track = []
        for each_note_row in shortened_track:
            n = each_note_row[0]
            ranges = np.array(each_note_row[1])
            print(ranges.shape)
            ranges[:, 0] = ranges[:, 0] / 3
            modified_track.append([n, ranges])
        return modified_track

    def request_handler(self):
        """Checks the requests in other classes request queues"""
        # print(self.instrument_panel.requests, self.window)

        if len(self.instrument_panel.requests) > 0:
            req = self.instrument_panel.requests.pop(0)
            print(req)
            if req == "instruments":
                if self.window == None:
                    self.window = AddInstrumentPanel()
                # print("Window:", self.window)

            elif type(req) == type({}):
                if "clicked" in req.keys():
                    instrument_selected = req["clicked"]
                    print("PProgramno",instrument_selected)
                    #self.pianoroll.add_track(instrument_selected)
                    self.pianoroll.selected_track = instrument_selected
                    self.events["update_pianoroll"] = True

        if self.window != None:
            if len(self.window.requests) > 0:
                req = self.window.requests.pop(0)
                if req == "close":
                    # Get the value returned from the window
                    self.instrument_panel.add_instrument(self.window.result)
                    self.pianoroll.add_track(self.window.result)
                    self.controls["update_ipanel"] = True
                    self.window = None

    def run(self):
        while not self.quit:
            self.draw()
            self.event_handler()
            self.play()
            self.request_handler()

        # print(self.pianoroll.notes)


if __name__ == '__main__':
    file = "data/again.mid"
    if len(sys.argv) == 2:
        file = sys.argv[1]

    d = Display(80)
    d.run()
