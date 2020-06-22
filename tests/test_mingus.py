from mingus.midi import fluidsynth
from mingus.containers.note import Note
from mingus.containers.note_container import NoteContainer
import time

fluidsynth.init("../assets/sound_fonts/Drama Piano.sf2", 'alsa')
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D#"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D#"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("B"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D"))
time.sleep(0.25)
fluidsynth.play_Note(Note("C"))
time.sleep(0.25)
fluidsynth.play_Note(Note("A"))
time.sleep(1)

fluidsynth.play_Note(Note("C"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("A"))
time.sleep(0.25)
fluidsynth.play_Note(Note("B"))
time.sleep(1)

fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("G#"))
time.sleep(0.25)
fluidsynth.play_Note(Note("B"))
time.sleep(0.25)
fluidsynth.play_Note(Note("C"))
time.sleep(1.0)

fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D#"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D#"))
time.sleep(0.25)
fluidsynth.play_Note(Note("E"))
time.sleep(0.25)
fluidsynth.play_Note(Note("B"))
time.sleep(0.25)
fluidsynth.play_Note(Note("D"))
time.sleep(0.25)
fluidsynth.play_Note(Note("C"))
time.sleep(0.25)
fluidsynth.play_Note(Note("A"))
time.sleep(1)


# Multiple notes at once
#fluidsynth.play_NoteContainer(NoteContainer(["C", "E"]))
#time.sleep(0.25)