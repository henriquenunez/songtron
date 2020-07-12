from mingus.midi import fluidsynth
from mingus.containers.note import Note
from mingus.containers.note_container import NoteContainer
import time

class NotePlayer():
    def __init__(self):
        self.notes = [["E",4], ["D#",4], ["E",4], ["D#",4], ["E",4], ["B",4], ["D",4], ["C",4], ["A",4]]
        self.songFont = "../assets/sound_fonts/Drama Piano.sf2"
        self.driver = "alsa"
        fluidsynth.init(self.songFont, self.driver)

    def setNotes(self, notes):
        self.notes = notes

    def playNotes(self):
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
            noteTone = note[0]
            noteType = note[1]

            fluidsynth.play_Note(Note(noteTone))
            time.sleep(1/float(noteType))

np = NotePlayer()
np.playNotes()
