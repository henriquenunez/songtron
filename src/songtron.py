import imageio
from sheet_music import Sheet_music
from classify_notes import Classify_notes
from tone_detector import Tone_detector
from note_player import Note_player

class Songtron():
    def __init__(self, filename):
        self.pic = imageio.imread(filename)
        self.sm = Sheet_music(self.pic)
        bboxes = self.sm.get_bboxes()
        lines = self.sm.get_lines_coord()
        self.cn = Classify_notes(self.sm.get_binarized(), bboxes)
        notes = self.cn.get_notes()
        cleff = self.cn.get_cleff()

        self.td = Tone_detector(lines, cleff, 60/101.0)
        self.td.set_notes(notes)
        notes_to_play = self.td.get_notes()

        print(notes_to_play)
        self.np = Note_player()
        self.np.set_notes(notes_to_play)
        self.np.play_notes()

songtron = Songtron("../assets/littleStar.png")
