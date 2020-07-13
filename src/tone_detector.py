import math
import numpy as np

#Class detects tone of the notes 
class Tone_detector():
    #h_lines = height of the lines
    #y_cleff = min and max y of the detected cleff
    #proportion = already calculated proportion of the cleff
    def __init__(self, h_lines, y_cleff, proportion):
        self.h_lines = h_lines
        self.y_cleff = y_cleff
        self.proportion = proportion
        self.notes = []
        print(self.h_lines, self.y_cleff)

    def detect_tones(self, center):
        #finding distances of each tone.
        self.distances = np.zeros(4)
        for i in range(4):
            self.distances[i] = self.h_lines[i+1]-self.h_lines[i]
        self.distance_tones = np.mean(self.distances)//2

        #finding the height of the note of reference from the cleff
        #not using this, using the coordinates of the second line
        #self.cleff = self.proportion * (self.y_cleff[1] - self.y_cleff[0]) + self.y_cleff[0]
        self.cleff = self.h_lines[3]

        print("dist:", self.distance_tones)
        print("cleff:", self.cleff)
        print("center:", center)
        #reference for cleff
        init_note = ord('G')
        init_number_note = 4

        h_diff = center - self.cleff
        qnt_tones = round(h_diff/self.distance_tones)
        tone = (init_note-65-qnt_tones) % 7

        init_number_note += math.floor((init_note-65-qnt_tones)/7)

        tone = chr(tone+65)
        print(tone, init_number_note)
        print()

        whole_tone = tone+'-'+str(init_number_note)
        return whole_tone

    #tempo of the note received by the classifier
    def detect_tempo(self, tempo):
        time = 0
        if(tempo == "semibreve"):
            time = 1
        elif(tempo == "minim"):
            time = 2
        elif(tempo == "crotchet"):
            time = 4
        elif(tempo == "quaver"):
            time = 8
        elif(tempo == "semiquaver"):
            time = 16
        elif(tempo == "demisemiquaver"):
            time = 32
        return time

    #set notes to self.notes
    def set_notes(self, info_notes):
        for info_note in info_notes:
            tone = self.detect_tones(info_note[0])
            tempo = self.detect_tempo(info_note[1])
            if tempo > 0:
                self.notes.append([tone, tempo])

    def get_notes(self):
        return self.notes

