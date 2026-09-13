#!/usr/bin/env python3

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.io import wavfile

# --------------------------------------------------------------------------
# Note <-> frequency math (12-tone equal temperament, A4 = 440 Hz)
# --------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
A4_FREQ = 440.0
A4_MIDI = 69  # MIDI note number for A4

FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
}


def note_to_freq(note: str) -> float:
    """'C4' -> 261.63, 'A4' -> 440.0, 'F#3' -> 92.5, 'Bb3' also works."""
    note = note.strip()
    # split into name + octave, e.g. "C#4" -> "C#", 4
    for i, ch in enumerate(note):
        if ch.isdigit() or (ch == "-" and i > 0):
            name, octave = note[:i], int(note[i:])
            break
    else:
        raise ValueError(f"Couldn't parse note '{note}' (missing octave number?)")

    name = FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name '{name}' in '{note}'")

    midi = NOTE_NAMES.index(name) + (octave + 1) * 12
    return A4_FREQ * (2 ** ((midi - A4_MIDI) / 12))


def freq_to_note(freq: float) -> Tuple[str, float]:
    """261.6 -> ('C4', cents_off_from_exact). cents_off is how far the input
    frequency was from the nearest in-tune note (100 cents = 1 semitone)."""
    if freq <= 0:
        raise ValueError("frequency must be positive")
    midi_float = A4_MIDI + 12 * np.log2(freq / A4_FREQ)
    midi = int(round(midi_float))
    cents_off = (midi_float - midi) * 100
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}", cents_off


def parse_dict_note_token(token: str) -> str:
    """A dictionary cell's note token might be a note name ('C4') or a raw
    frequency ('261.63'). Normalize both to a canonical note name."""
    token = token.strip()
    try:
        freq = float(token)
        name, _ = freq_to_note(freq)
        return name
    except ValueError:
        # not a bare number -> treat it as a note name, validate by round-tripping
        freq = note_to_freq(token)
        name, _ = freq_to_note(freq)
        return name


# --------------------------------------------------------------------------
# Dictionary loading
# --------------------------------------------------------------------------

class Dictionary:
    """Maps a frozenset/tuple of up to 3 note names to an English word."""

    def __init__(self, path: str):
        df = pd.read_excel(path)
        cols = {c.lower().strip(): c for c in df.columns}
        if "notes" not in cols or "word" not in cols:
            raise ValueError(
                f"Expected columns 'Notes' and 'Word' in {path}, found {list(df.columns)}"
            )
        notes_col, word_col = cols["notes"], cols["word"]

        self.lookup = {}
        self.max_chord_size = 1
        for _, row in df.iterrows():
            raw_notes, word = row[notes_col], row[word_col]
            if pd.isna(raw_notes) or pd.isna(word):
                continue
            tokens = [t for t in str(raw_notes).split(",") if t.strip()]
            if not (1 <= len(tokens) <= 3):
                raise ValueError(
                    f"Row '{raw_notes}' -> '{word}' has {len(tokens)} notes; only 1-3 supported"
                )
            names = tuple(sorted(parse_dict_note_token(t) for t in tokens))
            self.max_chord_size = max(self.max_chord_size, len(names))
            if names in self.lookup:
                print(
                    f"[warning] duplicate chord {names} maps to both "
                    f"'{self.lookup[names]}' and '{word}'; keeping the first",
                    file=sys.stderr,
                )
                continue
            self.lookup[names] = str(word)

    def translate(self, note_names: List[str]) -> Optional[str]:
        key = tuple(sorted(set(note_names)))
        return self.lookup.get(key)

    def __len__(self):
        return len(self.lookup)


# --------------------------------------------------------------------------
# Pitch / chord detection from an audio buffer
# --------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    sample_rate: int = 44100
    max_notes: int = 3
    min_freq: float = 60.0        # ignore rumble below this
    max_freq: float = 4000.0      # ignore hiss above this
    peak_prominence_ratio: float = 0.15  # peak must be >= this * strongest peak
    min_note_separation_semitones: float = 0.8  # merge peaks closer than this
    silence_threshold: float = 0.01  # RMS below this = "no notes"


def detect_notes(samples: np.ndarray, config: DetectorConfig) -> List[str]:
    """Given a mono chunk of audio, return up to `max_notes` note names
    for the strongest simultaneous tones present, or [] if it's silent."""
    if len(samples) == 0:
        return []

    rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    if rms < config.silence_threshold:
        return []

    # Window to reduce spectral leakage, then FFT.
    windowed = samples * np.hanning(len(samples))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / config.sample_rate)

    band = (freqs >= config.min_freq) & (freqs <= config.max_freq)
    freqs, spectrum = freqs[band], spectrum[band]
    if len(spectrum) == 0 or spectrum.max() == 0:
        return []

    peak_idx, _ = find_peaks(spectrum, height=spectrum.max() * config.peak_prominence_ratio)
    if len(peak_idx) == 0:
        return []

    # Sort candidate peaks by strength, strongest first.
    peak_idx = sorted(peak_idx, key=lambda i: spectrum[i], reverse=True)

    chosen_freqs: List[float] = []
    for i in peak_idx:
        f = freqs[i]
        # skip if this is basically a semitone-duplicate of (or harmonic near)
        # a frequency we already picked
        too_close = any(
            abs(12 * np.log2(f / cf)) < config.min_note_separation_semitones
            for cf in chosen_freqs
        )
        if not too_close:
            chosen_freqs.append(f)
        if len(chosen_freqs) == config.max_notes:
            break

    names = [freq_to_note(f)[0] for f in chosen_freqs]
    return sorted(set(names))


# --------------------------------------------------------------------------
# File mode
# --------------------------------------------------------------------------

def run_file_mode(wav_path: str, dictionary: Dictionary, config: DetectorConfig,
                   window_seconds: float, verbose: bool = True) -> List[Tuple[float, List[str], Optional[str]]]:
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    data = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.floating) and np.abs(data).max() > 1.0:
        data /= np.abs(data).max()
    elif data.dtype != np.float64 or np.abs(data).max() > 1.0:
        data /= 32768.0 if np.abs(data).max() > 1.0 else 1.0

    config.sample_rate = sr
    window_len = int(window_seconds * sr)
    results = []
    for start in range(0, len(data), window_len):
        chunk = data[start:start + window_len]
        if len(chunk) < window_len // 2:
            break
        notes = detect_notes(chunk, config)
        word = dictionary.translate(notes) if notes else None
        t = start / sr
        results.append((t, notes, word))
        if verbose:
            label = word if word else ("..." if notes else "(silence)")
            notes_str = ", ".join(notes) if notes else "-"
            print(f"[{t:6.2f}s] notes={notes_str:<20} -> {label}")
    return results


# --------------------------------------------------------------------------
# Live mode (optional dependency: sounddevice)
# --------------------------------------------------------------------------

def run_live_mode(dictionary: Dictionary, config: DetectorConfig, window_seconds: float,
                   debug: bool = False):
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Live mode needs the 'sounddevice' package.\n"
            "Install it locally with:  pip install sounddevice\n"
            "(This also requires PortAudio and a working microphone -- it won't "
            "run in a headless sandbox.)",
            file=sys.stderr,
        )
        sys.exit(1)

    device_info = sd.query_devices(sd.default.device[0], "input")
    print(f"Using input device: {device_info['name']}")
    print("Listening... (Ctrl+C to stop)")
    if debug:
        print("[debug mode] printing every window's RMS level and detected notes\n")

    last_word = None
    window_len = int(window_seconds * config.sample_rate)

    def callback(indata, frames, time_info, status):
        nonlocal last_word
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0]
        rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
        notes = detect_notes(mono, config)
        word = dictionary.translate(notes) if notes else None

        if debug:
            notes_str = ", ".join(notes) if notes else "-"
            match_str = word if word else "(no dictionary match)" if notes else "(below silence threshold)"
            print(f"rms={rms:.4f}  notes={notes_str:<15} -> {match_str}")
        elif word and word != last_word:
            print(f"-> {word}   (notes: {', '.join(notes)})")

        last_word = word if notes else None

    with sd.InputStream(
        channels=1,
        samplerate=config.sample_rate,
        blocksize=window_len,
        callback=callback,
    ):
        while True:
            time.sleep(0.1)


# --------------------------------------------------------------------------
# Self-test: synthesize each dictionary chord and confirm it decodes back
# --------------------------------------------------------------------------

def run_selftest(dict_path: str, config: DetectorConfig, window_seconds: float = 0.5):
    dictionary = Dictionary(dict_path)
    sr = config.sample_rate
    n_samples = int(window_seconds * sr)
    t = np.arange(n_samples) / sr

    passed, failed = 0, 0
    for chord, word in dictionary.lookup.items():
        signal = np.zeros(n_samples)
        for note in chord:
            freq = note_to_freq(note)
            signal += np.sin(2 * np.pi * freq * t)
        signal /= len(chord)
        signal += np.random.normal(0, 0.01, size=signal.shape)  # a little noise

        detected = detect_notes(signal, config)
        got_word = dictionary.translate(detected)
        ok = got_word == word
        passed += ok
        failed += not ok
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] chord={chord} -> expected '{word}', detected notes={detected}, got '{got_word}'")

    print(f"\n{passed} passed, {failed} failed out of {len(dictionary)} dictionary entries.")
    return failed == 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Translate musical notes/chords into English words.")
    ap.add_argument("--dict", default="dictionary.xlsx", help="Path to the .xlsx dictionary")
    ap.add_argument("--mode", choices=["file", "live"], help="file: analyze a .wav; live: use microphone")
    ap.add_argument("--input", help="Path to a .wav file (required for --mode file)")
    ap.add_argument("--window", type=float, default=0.5, help="Analysis window size in seconds")
    ap.add_argument("--threshold", type=float, default=0.15,
                     help="Peak prominence ratio (0-1); lower = more sensitive/noisier")
    ap.add_argument("--selftest", action="store_true", help="Synthesize and verify every dictionary entry")
    ap.add_argument("--debug", action="store_true",
                     help="Live mode: print RMS level and detected notes for every window, even unmatched ones")
    args = ap.parse_args()

    config = DetectorConfig(peak_prominence_ratio=args.threshold)

    if args.selftest:
        ok = run_selftest(args.dict, config, window_seconds=args.window)
        sys.exit(0 if ok else 1)

    dictionary = Dictionary(args.dict)
    print(f"Loaded {len(dictionary)} chord(s) from {args.dict}")

    if args.mode == "file":
        if not args.input:
            ap.error("--mode file requires --input path/to/audio.wav")
        run_file_mode(args.input, dictionary, config, args.window)
    elif args.mode == "live":
        run_live_mode(dictionary, config, args.window, debug=args.debug)
    else:
        ap.error("Specify --mode file or --mode live (or use --selftest)")


if __name__ == "__main__":
    main()