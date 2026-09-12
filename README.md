# project-hail-mary-eridian-translator
A Project Hail Mary inspired audio to text translator that uses FFT-based frequency analysis to decode sets of up to 3 musical notes into words using a customizable dictionary.

Claude Sonnet 5 was used for debugging help, as well as documentation and cleaning up the code for clearer structure and presentation.

## Files

- `note_translator.py` — the program
- `dictionary.xlsx` — example dictionary (7 sample words) — edit this, or make your own
- `test_message.wav` — a synthetic test clip that plays "hello, yes, friend, goodbye"

## Setup

```
pip install numpy scipy pandas openpyxl
# only needed for live microphone mode:
pip install sounddevice
```

## The dictionary format

An `.xlsx` file with two columns, `Notes` and `Word`:

| Notes         | Word     |
|---------------|----------|
| C4            | hello    |
| C4, E4        | yes      |
| C4, E4, G4    | friend   |

- 1 to 3 notes per row, comma-separated.
- Notes can be note names (`C4`, `F#5`, `Bb3`) or raw Hz (`261.63`) — mix and match if you like.
- Order doesn't matter — `C4, E4` and `E4, C4` are the same chord.
- Octave matters — `C4` (middle C) ≠ `C5`.

## Running it

**Self-test** (no audio hardware needed — synthesizes every dictionary entry and checks it decodes correctly):
```
python note_translator.py --dict dictionary.xlsx --selftest
```

**Decode a .wav file**, analyzing it in fixed-size windows:
```
python note_translator.py --dict dictionary.xlsx --mode file --input test_message.wav --window 0.5
```

**Live microphone** (run this on your own machine, not in a sandbox):
```
python note_translator.py --dict dictionary.xlsx --mode live --window 0.5
```
Play a chord near your mic and it'll print the matching word.

**Debug mode**, run this for the program to output what notes it detects, if any, every decoding loop:
```
python note_translator.py --dict dictionary.xlsx --mode live --window 0.5 --debug
```

## How it works

1. Audio is split into fixed-length windows (`--window`, default 0.5s).
2. Each window is FFT'd to get its frequency spectrum.
3. The strongest peaks in the spectrum are picked out (up to 3), skipping anything within ~0.8 semitones of a peak already picked, so a single note's own overtones don't get mistaken for extra "notes."
4. Each peak frequency is converted to the nearest equal-tempered note name (A4 = 440 Hz).
5. The set of note names is looked up in your dictionary.

## Tuning knobs

- `--window`: bigger = better frequency resolution (can tell close notes apart) but slower to react and blurs together fast note changes. 0.3–1.0s is a reasonable range for held chords.
- `--threshold`: how prominent a peak must be (as a fraction of the strongest peak) to count as a "note." Lower it if quiet notes in a chord aren't being detected; raise it if you're picking up noise/harmonics as extra phantom notes.

## Limitations

- **Real instruments are messy.** Real instruments like a guitar or piano have overtones (harmonics) when played. The synthetic self-test uses pure sine tones, which is the easy case. Real audio may need threshold tuning, or it may be easier to "play" a frequency generator online.
- **Window boundaries**: if a chord changes right in the middle of an analysis window, that window can pick up a blend of both chords and fail to match anything. Shorter windows reduce this at the cost of frequency resolution.
