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
        self.controls = {"fast_move_piano_roll":False ,"h_zoom": 1, "timebar": 0, "bar_move_velocity": 0, "make_velocity_0": True, "nthnote": 2}
        self.measures = measures
        py.font.init()
        self.notes = np.array(np.ones((48, self.measures * 32))) * -1
        self.notes_index = []
        # print(self.notes)
        self.timeFont = Font(10)
        self.update()
        self.nkeys = 48

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

    def play_notes(self, show_map=False):
        """
        Converts the notes array to pypianoroll format
        and then using MultiTrack object, the midi file
        is written to the file system
        :return:
        """
        pianoroll = np.zeros((self.notes.shape[1] * 3, 128))

        for i in range(self.notes.shape[0]):
            note_track = np.array(self.notes[i, :])
            note_track += 1
            # note_track[self.in1d_alternative_2D(note_track, np.array([-1]))]=0
            notes_pos = np.nonzero(note_track)

            # print("Notepos",note_track)
            # print("No",)
            f = 3
            for pos in notes_pos[0]:
                # if pos*f+(2**note_track[pos])*f < pianoroll.shape[0]:
                # print("Error",pos*f,pos*f+(2**(note_track[pos]-1))*f-1,83-i)
                # pianoroll[int(pos*f):int(pos*f+(2**(note_track[pos])-1)*f)-1,83-i] = velocity
                print("Error", f * pos, f * pos + self.dura_to_timecell(note_track[pos] - 1),
                      self.dura_to_timecell(note_track[pos] - 1))
                pianoroll[f * pos:f * pos + self.dura_to_timecell(note_track[pos] - 1) + 1, 83 - i] = 90

        # for i in range()
        #    for pos in notes_pos[0]:
        #        print(pianoroll[pos*3:int(pos*3+(2**(note_track[pos])-1)*3)-1,83-i])
        print("pianoroll")
        # y = pianoroll.T
        print(self.notes_index)
        # print(y)
        # for i in y:
        #    print(i)
        tracker = Track(pianoroll=pianoroll)
        multitrack = Multitrack(tracks=[tracker])
        multitrack.write("create1.mid")

        if show_map:
            tracker.plot()
            plt.show()

    def print_notes(self):
        """
        Converts the numpy array of the values in
        the pianoroll to midi file and save it to filesystem
        :return:
        """
        music = []
        scale = ['B', 'A#', 'A', 'G#', 'G', 'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']
        dura = [0.125, 0.25, 0.5, 1, 2, 4]
        for i in range(len(self.notes[0])):
            notes_at_i = []

            for j in range(len(self.notes[:, i])):
                if self.notes[j, i] != -1:
                    notes_at_i.append([j, self.notes[j, i], i])
                    # print("i value",i)
            if len(notes_at_i) != 0:
                music.append(notes_at_i)
        streamer = stream.Stream()
        start = 0

        print("Music", music)
        for nota in self.notes_index:
            print("N", nota, self.notes[nota[0]][nota[1]])

        for _chord in music:
            j = []
            done = False
            for notes in _chord:
                # print(notes)
                sect = int(notes[0] / 12)
                key = scale[notes[0] % 12] + str(5 - sect)
                keynote = note.Note(key)
                keynote.duration = duration.Duration(dura[int(notes[1])])
                keynote.offset = notes[2] * 0.125
                # print("keynote.duration",dura[int(notes[1])])
                j.append(keynote)
                if done == False:
                    if start == notes[2]:
                        start += dura[int(notes[1])] * 8
                        # print(start)
                    else:
                        # print("Difference Calculation",notes[2],"-",start)
                        diff = notes[2] - start
                        start += diff + dura[int(notes[1])] * 8
                        # print("Difference",diff)
                        for u in range(int(diff)):
                            r = note.Rest()
                            r.duration = duration.Duration(0.125)
                            streamer.append(r)
                    done = True

            music_chord = chord.Chord(j)
            streamer.append(music_chord)
        streamer.write('midi', fp='create1.midi')
        # streamer.show()
        # print()
        # for thisNote in streamer.getElementsByClass(chord.Chord):
        # print(thisNote,thisNote.offset)

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
        if self.notes[k, s] != -1:
            self.notes[k, s] = -1
            self.notes_index.remove([k, s])

    def enternote(self, s, k):
        # print(self.notes[k,s])
        if s < self.notes.shape[1] and self.notes[k, s] != -1:
            self.notes[k, s] = -1
            self.notes_index.remove([k, s])
        elif s < self.notes.shape[1]:
            self.notes[k, s] = self.controls["nthnote"]
            self.notes_index.append([k, s])
        # print(self.notes)

    def draw_notes(self):
        for y, x in self.notes_index:
            # print(y,x,self.notes.shape[1])
            j = (x * 20 * self.controls["h_zoom"] + self.controls["timebar"]) + 20
            # print((x*20*self.controls["h_zoom"]+self.controls["timebar"])+20)
            if j + 20 * self.controls["h_zoom"] * (2 ** self.notes[y][x]) - 2 >= 20 and j < 800:
                py.draw.rect(self.pianoroll, (200, 0, 0), (
                (x * 20 * self.controls["h_zoom"] + self.controls["timebar"]) + 20, y * 10 + 10 + 2,
                20 * self.controls["h_zoom"] * (2 ** self.notes[y][x]) - 2, 10 - 2))

    def send_event(self, event):
        if event == "measure_increased":
            self.pnotes = np.array(np.ones((48, self.measures * 32))) * -1
            self.pnotes[:self.notes.shape[0], :self.notes.shape[1]] = self.notes
            self.notes = self.pnotes

        elif event == "measure_decreased":
            self.pnotes = np.array(np.ones((48, self.measures * 32))) * -1
            self.pnotes = self.notes[:self.pnotes.shape[0], :self.pnotes.shape[1]]
            self.notes = self.pnotes

            t = list(self.notes_index)
            for x, y in t:
                # print(y,self.notes.shape[1])
                if y >= self.notes.shape[1]:
                    self.notes_index.remove([x, y])
                    # print("removed a note",x,y)
            # for x,y in self.notes_index:
            # print("A",x,y)
            pass

    def load_file(self, parsed_track, measure_limit):
        """
        Converts the parsed track and loads into the pianoroll
        numpy array. parsed_track is divided by 3 so the basic time
        unit is 32nd note.
        """
        # print("Par",parsed_track )
        # print()
        array = np.ones([48, int(measure_limit)]) * -1
        for note_index, notes_track in parsed_track:
            note_index = note_index - 36
            if note_index >= 0 and note_index < 48:
                for pos, dura in notes_track:
                    # print("Error",note_index,pos,np.log2(dura))
                    if int(pos) < array.shape[1]:
                        print(47 - note_index, int(pos), array.shape, np.log2(dura))
                        array[47 - note_index][int(pos)] = np.log2(dura)
                        self.notes_index.append([47 - note_index, int(pos)])
        self.notes = array
        # print("Nonzeros",np.nonzero(self.notes))
        print(self.notes_index)
        # for note in self.notes_index:
        # print("N",note,self.notes[note[0]][note[1]])
        # pass



