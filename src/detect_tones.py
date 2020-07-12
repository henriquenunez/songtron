import math

class tone_detector():

    #h_lines = height of the lines
    #y_cleff = min and max y of the detected cleff
    #proportion = already calculated proportion of the cleff
    def __init__(self, h_lines, y.cleff, proportion):
        self.h_lines = h_lines
        self.y.cleff = y.cleff
        self.proportion = proportion
        self.notes = []

    def detect_tones(self, center):
        #finding distances of each tone.
        self.distances = np.zeros(4)
        for i in range(4):
            self.distances[i] = self.h_lines[i+1]-self.h_lines[i]
        self.distance_tones = np.mean(self.distances)//2

        #finding the height of the note of reference from the cleff
        self.cleff = self.proportion * (self.y.cleff[1] - self.y.cleff[0]) + self.y.cleff[0]
        #reference for cleff
        init_note = ord('G')
        init_number_note = 3

        h_diff = center - self.cleff
        qnt_tones = round(h_diff/self.distance_tones)
        tone = (init_note-65-qnt_tones) % 7

        init_number_note += math.floor((init_note-65-qnt_tones)/7)

        tone = chr(tone+65)
        print(tone, init_number_note)

        whole_tone = tone+'-'+str(init_number_note)
        return whole_tone

    def detect_tempo(self, tempo):
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

    def set_notes(self, info_note):
        tone = self.detect_tones(info_note[0])
        tempo = self.detect_tempo(info_note[1])
        self.notes.append([tone, tempo])

    def get_notes(self):
        return self.notes
