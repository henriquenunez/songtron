from mingus.midi import fluidsynth
from mingus.containers.note import Note
from mingus.containers.note_container import NoteContainer
import time

class Note_player():
    def __init__(self):
        self.notes = [["E",4], ["D#",4], ["E",4], ["D#",4], ["E",4], ["B",4], ["D",4], ["C",4], ["A",4]]
        self.song_font = "../assets/sound_fonts/Drama Piano.sf2"
        self.driver = "alsa"
        fluidsynth.init(self.song_font, self.driver)

    def set_notes(self, notes):
        self.notes = notes

    def play_notes(self):
        '''
        The input is a vector of pairs
        input example: notes=[["E",4], ["D#",4], ["E",2]]

        HOW NOTES WORKS
        notes example =  [note, note, note, note]
        note = [noteTone, noteType]

        noteTone is a string, examples:
            "E", "D#", "A-4", "B-5", "Cb"

        noteType is a integer, types:
            1  -> Whole
            2  -> Half
            4  -> Quarter
            8  -> Eighth
            16 -> Sixteenth
        '''

        for note in self.notes:
            note_tone = note[0]
            note_type = note[1]

            fluidsynth.play_Note(Note(note_tone))
            time.sleep(1/float(note_type))

np = Note_player()
np.play_notes()
