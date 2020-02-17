import pygame as py
import pygame.gfxdraw
import numpy as np
from music21 import *
from font import *
from pypianoroll import Track, Multitrack
from matplotlib import pyplot as plt


class Pianoroll:
    def __init__(self, measures):
        self.height = 500
        self.width = 800

        self.pianoroll = py.Surface((self.width, self.height))
        self.controls = {"fast_move_piano_roll": False, "h_zoom": 1, "timebar": 0, "bar_move_velocity": 0,
                         "make_velocity_0": True, "nthnote": 2}
        self.measures = measures
        py.font.init()

        self.n_tracks = 1
        self.selected_track = 0
        self.notes = [np.array(np.ones((48, self.measures * 32))) * -1]
        self.programs = [0]
        self.notes_index = [[]]
        # print(self.notes)
        self.timeFont = Font(10)
        self.update()
        self.nkeys = 48

    def add_track(self, programno):
        self.programs.append(programno)
        self.notes.append(np.array(np.ones((48, self.measures * 32))) * -1)
        self.notes_index.append([])
        self.n_tracks += 1
        self.selected_track = self.n_tracks - 1

    def reduce_velocity(self):
        """
        This reduces the scrolling velocity in
        the pianoroll.
        :return:
        """
        if self.controls["make_velocity_0"]:
            # print(self.controls["bar_move_velocity"])
            self.controls["bar_move_velocity"] = 0

    def in1d_alternative_2D(self, nparr, arr):
        idx = np.searchsorted(arr, nparr.ravel())
        idx[idx == len(arr)] = 0
        return arr[idx].reshape(nparr.shape) == nparr

    def dura_to_timecell(self, x):
        return {0: 2, 1: 4, 2: 10, 3: 21, 4: 44, 5: 90}[x]

    def play_single_array(self, trackno, trackpos):
        """Code should be written"""
        pass

    def play_notes(self, show_map=False):
        """
        Converts the notes array to pypianoroll format
        and then using MultiTrack object, the midi file
        is written to the file system
        :return:
        """
        print(self.programs)
        pianorolls = []
        for track in self.notes:
            pianoroll = np.zeros((track.shape[1] * 3, 128))

            for i in range(track.shape[0]):
                note_track = np.array(track[i, :])
                note_track += 1

                notes_pos = np.nonzero(note_track)
                f = 3
                for pos in notes_pos[0]:
                    print("Error", f * pos, f * pos + self.dura_to_timecell(note_track[pos] - 1),
                          self.dura_to_timecell(note_track[pos] - 1))
                    pianoroll[f * pos:f * pos + self.dura_to_timecell(note_track[pos] - 1) + 1, 83 - i] = 90
            pianorolls.append(pianoroll)

        # print("pianoroll")
        print(self.notes_index)
        tracks = []
        for i in range(len(pianorolls)):
            tracker = Track(pianoroll=pianorolls[i],program=self.programs[i])
            tracks.append(tracker)
        multitrack = Multitrack(tracks=tracks)
        multitrack.write("create1.mid")

        if show_map:
            print("Show map will not work")
            # tracker.plot()
            # plt.show()

    def update(self):
        """
        Updates the pianoroll surface draws such as the
        vertical lines, numberline, notes.
        :return:
        """
        roll_index = 0
        key_color = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        key_color.reverse()
        self.pianoroll.fill((30, 30, 40))
        self.reduce_velocity()

        if self.controls["bar_move_velocity"] > 0:
            self.controls["timebar"] -= self.controls["bar_move_velocity"]
        elif self.controls["timebar"] < 0:
            self.controls["timebar"] -= self.controls["bar_move_velocity"]

        # draw horizontal lines
        for j in range(10, self.height, 10):
            py.gfxdraw.line(self.pianoroll, 20, j, self.width, j, (255, 255, 255, 30))

            # below code is for rendering the piano keys on the left
            if key_color[roll_index] == 0:
                py.draw.rect(self.pianoroll, (255, 255, 255), (0, j + 2, 15, 8))  # white keys
                py.draw.rect(self.pianoroll, (0, 100, 50), (20, j + 2, self.width - 20, 8))
            else:
                py.draw.rect(self.pianoroll, (0, 0, 0), (0, j + 2, 15, 8))  #
                pass
            roll_index = (roll_index + 1) % 12

        roll_index = 0
        self.draw_notes()
        for j in range(10, self.height, 10):
            if (key_color[roll_index] == 0):
                py.draw.rect(self.pianoroll, (255, 255, 255), (0, j + 2, 15, 8))  # white keys
                # py.draw.rect(self.pianoroll, (30, 30, 50), (20, j + 2, self.width - 20, 8))
            else:
                py.draw.rect(self.pianoroll, (0, 0, 0), (0, j + 2, 15, 8))  #
                pass
            roll_index = (roll_index + 1) % 12

        # draw vertical lines
        fourth = 0

        # t = int(np.ceil( (self.controls["timebar"]+20)/int(20 * self.controls["h_zoom"])))
        # s = t+800
        for i in range(20, int(20 * self.controls["h_zoom"] * self.measures * 32) + 30,
                       int(20 * self.controls["h_zoom"])):
            # the two conditions are for making the fourth line brighter
            # q = str(int(self.controls["timebar"]))+" "+str(i)+" "+str(int(self.controls["timebar"] + i))
            if int(self.controls["timebar"] + i) >= 20 and int(self.controls["timebar"] + i) <= 800:
                # q+= str(" D")
                if fourth % 4 == 0:
                    py.gfxdraw.line(self.pianoroll, int(self.controls["timebar"] + i), 10,
                                    int(self.controls["timebar"] + i), self.height - 10, (255, 255, 255, 80))
                else:
                    py.gfxdraw.line(self.pianoroll, int(self.controls["timebar"] + i), 10,
                                    int(self.controls["timebar"] + i), self.height - 10, (255, 255, 255, 30))
            # print(q)
            fourth += 1

        # creates color differentiation for the black key rows and white key rows

        # creates the timebar on the top of pianoroll
        index = 0
        for i in range(20, int(20 * self.controls["h_zoom"] * (self.measures + 1) * 32),
                       int(20 * self.controls["h_zoom"] * 32)):
            surf, rect = self.timeFont.text_object(str(index))
            rect.topleft = (self.controls["timebar"] + i, 0)
            if (rect.topleft[0] >= 20):
                self.pianoroll.blit(surf, rect)
            index += 1

    def deletenote(self, s, k):
        if self.notes[self.selected_track][k, s] != -1:
            self.notes[self.selected_track][k, s] = -1
            self.notes_index[self.selected_track].remove([k, s])

    def enternote(self, s, k):
        if s < self.notes[self.selected_track].shape[1] and self.notes[self.selected_track][k, s] != -1:
            self.notes[self.selected_track][k, s] = -1
            self.notes_index[self.selected_track].remove([k, s])
        elif s < self.notes[self.selected_track].shape[1]:
            self.notes[self.selected_track][k, s] = self.controls["nthnote"]
            self.notes_index[self.selected_track].append([k, s])

    def draw_notes(self):
        print(len(self.notes_index))
        if len(self.notes_index) > 0:
            for y, x in self.notes_index[self.selected_track]:
                j = (x * 20 * self.controls["h_zoom"] + self.controls["timebar"]) + 20
                if j + 20 * self.controls["h_zoom"] * (2 ** self.notes[self.selected_track][y][x]) - 2 >= 20 and j < 800:
                    py.draw.rect(self.pianoroll, (200, 0, 0), (
                        (x * 20 * self.controls["h_zoom"] + self.controls["timebar"]) + 20, y * 10 + 10 + 2,
                        20 * self.controls["h_zoom"] * (2 ** self.notes[self.selected_track][y][x]) - 2, 10 - 2))

    def send_event(self, event):
        if event == "measure_increased":
            # Error: Possibile if note=pnotes doesn't change the value in self.notes
            for note in self.notes:
                pnotes = np.array(np.ones((48, self.measures * 32))) * -1
                pnotes[:note.shape[0], :note.shape[1]] = note
                note = pnotes

        elif event == "measure_decreased":
            for i in range(len(self.notes)):
                pnotes = np.array(np.ones((48, self.measures * 32))) * -1
                pnotes = self.notes[i][:pnotes.shape[0], :pnotes.shape[1]]
                self.notes[i] = pnotes

                t = list(self.notes_index[i])
                for x, y in t:
                    if y >= self.notes[i].shape[1]:
                        self.notes_index[self.notes.index(self.notes[i])].remove([x, y])

    def load_file(self, parsed_track, measure_limit):
        """
        Converts the parsed track and loads into the pianoroll
        numpy array. parsed_track is divided by 3 so the basic time
        unit is 32nd note.
        """
        array = np.ones([48, int(measure_limit)]) * -1
        for note_index, notes_track in parsed_track:
            note_index = note_index - 36
            if note_index >= 0 and note_index < 48:
                for pos, dura in notes_track:
                    if int(pos) < array.shape[1]:
                        print(47 - note_index, int(pos), array.shape, np.log2(dura))
                        array[47 - note_index][int(pos)] = np.log2(dura)
                        self.notes_index.append([47 - note_index, int(pos)])
        self.notes = array
        print(self.notes_index)

    def clear_current_track(self):
        self.notes[self.selected_track] = self.notes[self.selected_track] * 0
        self.notes[self.selected_track] = self.notes[self.selected_track] - 1
        self.notes_index[self.selected_track] = []
        print("Cleared")