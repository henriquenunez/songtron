import imageio
from sheet_music import Sheet_music
from classify_notes import Classify_notes
from tone_detector import Tone_detector
from note_player import Note_player

class Songtron():
    def __init__(self, filename):
        self.pic = imageio.imread(filename)
        self.sm = Sheet_music(self.pic)
        self.bboxes = self.sm.get_bboxes()
        self.cn = Classify_notes(self.pic, self.bboxes)

songtron = Songtron("../assets/dataset4.png")
