import argparse
import json
import os
import re
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from music21 import chord as m21chord
from music21 import converter, metadata, note, pitch as m21pitch, stream
from music21 import interval as m21interval

SUPPORTED_EXTENSIONS = {".mid", ".midi", ".xml", ".musicxml", ".mxl"}
ALLOWED_CHORD_QUALITIES = {"maj", "min", "7", "maj7", "min7", "dim"}
CHORD_DURATION_GRID = [1.0, 2.0, 4.0]
CHORD_BAR_LENGTH = 4.0
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
PITCH_CLASS_MAP = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D-": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E-": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G-": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A-": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B-": 10,
    "B": 11,
    "CB": 11,
}
VOICE_RANGES = {
    "bass": (36, 60),
    "tenor": (45, 67),
    "alto": (52, 76),
    "soprano": (60, 84),
}
VOICE_MAX_JUMP = {
    "bass": 7,
    "tenor": 7,
    "alto": 7,
    "soprano": 7,
}
VOICE_ORDER = ["bass", "tenor", "alto", "soprano"]
CHORALE_REPEAT_NOTE_BASE_PENALTY = 5.0
CHORALE_REPEAT_NOTE_STREAK_PENALTY = 2.0
DEFAULT_MELODY_INPUT_FOLDER = "melodyInput"
ROMAN_INTERVALS_MAJOR = {"i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11}
ROMAN_INTERVALS_MINOR = {"i": 0, "ii": 2, "iii": 3, "iv": 5, "v": 7, "vi": 8, "vii": 10}


class SecondOrderMarkov:
    """Generic second-order Markov model with Laplace smoothing."""

    def __init__(self, states, laplace_alpha=1.0):
        if laplace_alpha < 0:
            raise ValueError("laplace_alpha must be >= 0")
        if not states:
            raise ValueError("states must not be empty")

        self.states = list(states)
        self.laplace_alpha = laplace_alpha
        self.num_states = len(self.states)
        self.initial_pair_probabilities = np.zeros((self.num_states, self.num_states))
        self.transition_matrix = np.zeros(
            (self.num_states, self.num_states, self.num_states)
        )
        self._state_indexes = {state: i for i, state in enumerate(self.states)}

    def train(self, sequence):
        if len(sequence) < 2:
            raise ValueError("Need at least 2 items to train a second-order model")

        self.initial_pair_probabilities.fill(0)
        self.transition_matrix.fill(0)

        for i in range(len(sequence) - 1):
            a = self._state_indexes[sequence[i]]
            b = self._state_indexes[sequence[i + 1]]
            self.initial_pair_probabilities[a, b] += 1

        total = self.initial_pair_probabilities.sum()
        if total:
            self.initial_pair_probabilities /= total
        else:
            self.initial_pair_probabilities = np.full(
                (self.num_states, self.num_states),
                1 / (self.num_states * self.num_states),
            )

        for i in range(len(sequence) - 2):
            a = self._state_indexes[sequence[i]]
            b = self._state_indexes[sequence[i + 1]]
            c = self._state_indexes[sequence[i + 2]]
            self.transition_matrix[a, b, c] += 1

        self._normalize_transition_matrix()

    def generate(self, length):
        if length <= 0:
            return []
        if length == 1:
            a, _ = self._generate_starting_pair()
            return [a]

        first, second = self._generate_starting_pair()
        out = [first, second]
        for _ in range(2, length):
            out.append(self._generate_next(out[-2], out[-1]))
        return out

    def _normalize_transition_matrix(self):
        smoothed = self.transition_matrix + self.laplace_alpha
        sums = smoothed.sum(axis=2, keepdims=True)
        uniform = np.full_like(smoothed, 1 / self.num_states)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.transition_matrix = np.divide(
                smoothed,
                sums,
                out=uniform,
                where=sums != 0,
            )

    def _generate_starting_pair(self):
        flat_probs = self.initial_pair_probabilities.ravel()
        flat_idx = np.random.choice(self.num_states * self.num_states, p=flat_probs)
        i, j = divmod(flat_idx, self.num_states)
        return self.states[i], self.states[j]

    def _generate_next(self, prev_state, curr_state):
        i = self._state_indexes[prev_state]
        j = self._state_indexes[curr_state]
        probs = self.transition_matrix[i, j]
        k = np.random.choice(self.num_states, p=probs)
        return self.states[k]


class FactorizedMelodyModel:
    """Models pitch intervals and durations separately, then combines them."""

    def __init__(self, interval_model, duration_model, seed_pitches):
        if not seed_pitches:
            raise ValueError("seed_pitches must not be empty")
        self.interval_model = interval_model
        self.duration_model = duration_model
        self.seed_pitches = list(seed_pitches)

    @classmethod
    def train_from_events(cls, events, laplace_alpha=1.0):
        if len(events) < 3:
            raise ValueError("Need at least 3 notes for interval + duration modeling")

        midi_sequence = [m for m, _ in events]
        duration_sequence = [d for _, d in events]
        interval_sequence = [
            midi_sequence[i + 1] - midi_sequence[i]
            for i in range(len(midi_sequence) - 1)
        ]

        interval_states = sorted(set(interval_sequence))
        duration_states = sorted(set(duration_sequence))

        interval_model = SecondOrderMarkov(interval_states, laplace_alpha=laplace_alpha)
        duration_model = SecondOrderMarkov(duration_states, laplace_alpha=laplace_alpha)

        interval_model.train(interval_sequence)
        duration_model.train(duration_sequence)

        return cls(interval_model, duration_model, sorted(set(midi_sequence)))

    def generate(self, length, seed_midi=None):
        if length <= 0:
            return []

        durations = self.duration_model.generate(length)
        if length == 1:
            midi_value = seed_midi if seed_midi is not None else int(np.random.choice(self.seed_pitches))
            midi_value = max(0, min(127, int(midi_value)))
            return [(midi_value, durations[0])]

        intervals = self.interval_model.generate(length - 1)
        current_midi = int(np.random.choice(self.seed_pitches)) if seed_midi is None else int(seed_midi)

        melody = []
        current_midi = max(0, min(127, int(current_midi)))
        melody.append((current_midi, durations[0]))

        for i in range(1, length):
            current_midi += int(intervals[i - 1])
            current_midi = max(0, min(127, current_midi))
            melody.append((current_midi, durations[i]))

        return melody

    def generate_constrained(
        self,
        length,
        seed_midi=None,
        pitch_min=48,
        pitch_max=84,
        max_jump=7,
        scale_pitch_classes=None,
        duration_values=None,
    ):
        if pitch_min > pitch_max:
            raise ValueError("pitch_min must be <= pitch_max")
        if max_jump < 1:
            raise ValueError("max_jump must be >= 1")

        if duration_values:
            allowed_durations = sorted(set(float(d) for d in duration_values))
        else:
            allowed_durations = None

        durations = self.duration_model.generate(length)
        if allowed_durations is not None:
            durations = [_quantize_duration(d, allowed_durations) for d in durations]

        if length <= 0:
            return []

        if seed_midi is None:
            current_midi = int(np.random.choice(self.seed_pitches))
        else:
            current_midi = int(seed_midi)

        current_midi = max(pitch_min, min(pitch_max, current_midi))
        if scale_pitch_classes is not None:
            current_midi = _snap_midi_to_scale(current_midi, scale_pitch_classes)
            current_midi = max(pitch_min, min(pitch_max, current_midi))

        if length == 1:
            return [(current_midi, durations[0])]

        intervals = self.interval_model.generate(length - 1)
        melody = [(current_midi, durations[0])]

        for i in range(1, length):
            delta = int(intervals[i - 1])
            delta = max(-max_jump, min(max_jump, delta))
            current_midi += delta
            current_midi = max(pitch_min, min(pitch_max, current_midi))
            if scale_pitch_classes is not None:
                current_midi = _snap_midi_to_scale(current_midi, scale_pitch_classes)
                current_midi = max(pitch_min, min(pitch_max, current_midi))
            melody.append((current_midi, durations[i]))

        return melody

    def save(self, model_path):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            interval_states=np.array(self.interval_model.states, dtype=object),
            interval_laplace_alpha=np.array([self.interval_model.laplace_alpha]),
            interval_initial_pair_probabilities=self.interval_model.initial_pair_probabilities,
            interval_transition_matrix=self.interval_model.transition_matrix,
            duration_states=np.array(self.duration_model.states, dtype=object),
            duration_laplace_alpha=np.array([self.duration_model.laplace_alpha]),
            duration_initial_pair_probabilities=self.duration_model.initial_pair_probabilities,
            duration_transition_matrix=self.duration_model.transition_matrix,
            seed_pitches=np.array(self.seed_pitches, dtype=int),
        )

    @classmethod
    def load(cls, model_path):
        with np.load(model_path, allow_pickle=True) as data:
            interval_model = SecondOrderMarkov(
                [int(x) for x in data["interval_states"].tolist()],
                laplace_alpha=float(data["interval_laplace_alpha"][0]),
            )
            interval_model.initial_pair_probabilities = data[
                "interval_initial_pair_probabilities"
            ]
            interval_model.transition_matrix = data["interval_transition_matrix"]

            duration_model = SecondOrderMarkov(
                [float(x) for x in data["duration_states"].tolist()],
                laplace_alpha=float(data["duration_laplace_alpha"][0]),
            )
            duration_model.initial_pair_probabilities = data[
                "duration_initial_pair_probabilities"
            ]
            duration_model.transition_matrix = data["duration_transition_matrix"]

            seed_pitches = [int(x) for x in data["seed_pitches"].tolist()]

        return cls(interval_model, duration_model, seed_pitches)


class ChordMarkovModel:
    """Second-order Markov model over chord tokens."""

    def __init__(self, chord_model):
        self.chord_model = chord_model

    @classmethod
    def train_from_events(cls, chord_events, laplace_alpha=1.0):
        if len(chord_events) < 3:
            raise ValueError("Need at least 3 chord events for chord modeling")

        states = sorted(set(chord_events))
        model = SecondOrderMarkov(states, laplace_alpha=laplace_alpha)
        model.train(chord_events)
        return cls(model)

    def generate(
        self,
        length,
        duration_values=None,
        repeat_penalty=0.0,
        scale_pitch_classes=None,
    ):
        progression = self._generate_with_repeat_penalty(
            length,
            repeat_penalty=repeat_penalty,
            scale_pitch_classes=scale_pitch_classes,
        )
        if duration_values:
            allowed = sorted(set(float(d) for d in duration_values))
            progression = [
                (root_pc, quality, _quantize_duration(duration, allowed))
                for root_pc, quality, duration in progression
            ]
        return progression

    def _generate_with_repeat_penalty(
        self, length, repeat_penalty=0.0, scale_pitch_classes=None
    ):
        if length <= 0:
            return []
        if repeat_penalty <= 0 and scale_pitch_classes is None:
            return self.chord_model.generate(length)

        repeat_penalty = max(0.0, min(0.99, float(repeat_penalty)))
        keep_factor = 1.0 - repeat_penalty

        allowed_mask = None
        if scale_pitch_classes is not None:
            allowed_mask = np.array(
                [
                    _chord_token_in_scale(token, scale_pitch_classes)
                    for token in self.chord_model.states
                ],
                dtype=bool,
            )
            if not np.any(allowed_mask):
                raise ValueError(
                    "No chord states fit the requested scale. "
                    "Try a different --key/--mode or retrain with more matching data."
                )

        initial_probs = self.chord_model.initial_pair_probabilities.copy()
        if allowed_mask is not None:
            initial_probs = initial_probs * allowed_mask[np.newaxis, :]
            initial_probs = initial_probs * allowed_mask[:, np.newaxis]
        initial_total = initial_probs.sum()
        if initial_total > 0:
            initial_probs = initial_probs / initial_total

        if length == 1:
            if initial_total > 0:
                flat_probs = initial_probs.ravel()
                flat_idx = np.random.choice(
                    self.chord_model.num_states * self.chord_model.num_states, p=flat_probs
                )
                first_idx, _ = divmod(flat_idx, self.chord_model.num_states)
            else:
                # Fallback when only isolated in-scale states exist but no observed pair.
                state_probs = (
                    self.chord_model.initial_pair_probabilities.sum(axis=1)
                    + self.chord_model.initial_pair_probabilities.sum(axis=0)
                )
                if allowed_mask is not None:
                    state_probs = state_probs * allowed_mask.astype(float)
                state_total = state_probs.sum()
                if state_total <= 0:
                    state_probs = allowed_mask.astype(float)
                    state_total = state_probs.sum()
                state_probs = state_probs / state_total
                first_idx = int(np.random.choice(self.chord_model.num_states, p=state_probs))
            first = self.chord_model.states[first_idx]
            return [first]

        if initial_total > 0:
            flat_probs = initial_probs.ravel()
            flat_idx = np.random.choice(
                self.chord_model.num_states * self.chord_model.num_states, p=flat_probs
            )
            first_idx, second_idx = divmod(flat_idx, self.chord_model.num_states)
            first = self.chord_model.states[first_idx]
            second = self.chord_model.states[second_idx]
        else:
            # Fallback when no in-scale pair appears in training starts.
            state_probs = (
                self.chord_model.initial_pair_probabilities.sum(axis=1)
                + self.chord_model.initial_pair_probabilities.sum(axis=0)
            )
            if allowed_mask is not None:
                state_probs = state_probs * allowed_mask.astype(float)
            state_total = state_probs.sum()
            if state_total <= 0:
                state_probs = allowed_mask.astype(float)
                state_total = state_probs.sum()
            state_probs = state_probs / state_total
            first_idx = int(np.random.choice(self.chord_model.num_states, p=state_probs))
            first = self.chord_model.states[first_idx]

            next_probs = self.chord_model.transition_matrix[first_idx, first_idx].copy()
            if allowed_mask is not None:
                next_probs[~allowed_mask] = 0.0
            next_total = next_probs.sum()
            if next_total <= 0:
                next_probs = state_probs
            else:
                next_probs = next_probs / next_total
            second_idx = int(np.random.choice(self.chord_model.num_states, p=next_probs))
            second = self.chord_model.states[second_idx]
        out = [first, second]

        def chord_identity(token):
            return (int(token[0]), str(token[1]))

        repeat_streak = 2 if chord_identity(first) == chord_identity(second) else 1

        for _ in range(2, length):
            prev_state = out[-2]
            curr_state = out[-1]
            i = self.chord_model._state_indexes[prev_state]
            j = self.chord_model._state_indexes[curr_state]
            probs = self.chord_model.transition_matrix[i, j].copy()

            curr_identity = chord_identity(curr_state)
            for idx, candidate in enumerate(self.chord_model.states):
                if chord_identity(candidate) == curr_identity:
                    probs[idx] *= keep_factor**repeat_streak
                if allowed_mask is not None and not allowed_mask[idx]:
                    probs[idx] = 0.0

            total = probs.sum()
            if total <= 0:
                probs = self.chord_model.transition_matrix[i, j].copy()
                if allowed_mask is not None:
                    probs[~allowed_mask] = 0.0
                    total = probs.sum()
                    if total <= 0:
                        probs = allowed_mask.astype(float)
                        probs = probs / probs.sum()
                    else:
                        probs = probs / total
            else:
                probs = probs / total

            next_idx = np.random.choice(self.chord_model.num_states, p=probs)
            next_state = self.chord_model.states[next_idx]
            out.append(next_state)

            if chord_identity(next_state) == curr_identity:
                repeat_streak += 1
            else:
                repeat_streak = 1

        return out

    def save(self, model_path):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            chord_states=np.array(self.chord_model.states, dtype=object),
            chord_laplace_alpha=np.array([self.chord_model.laplace_alpha]),
            chord_initial_pair_probabilities=self.chord_model.initial_pair_probabilities,
            chord_transition_matrix=self.chord_model.transition_matrix,
        )

    @classmethod
    def load(cls, model_path):
        with np.load(model_path, allow_pickle=True) as data:
            states = [tuple(s) for s in data["chord_states"].tolist()]
            model = SecondOrderMarkov(
                states,
                laplace_alpha=float(data["chord_laplace_alpha"][0]),
            )
            model.initial_pair_probabilities = data["chord_initial_pair_probabilities"]
            model.transition_matrix = data["chord_transition_matrix"]
        return cls(model)


def _load_cache(cache_path):
    if not Path(cache_path).exists():
        return {"files": {}}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            return data
    except Exception:
        pass
    return {"files": {}}


def _save_cache(cache_path, cache_data):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f)


def create_training_data_events():
    twinkle = [
        ("C5", 1),
        ("C5", 1),
        ("G5", 1),
        ("G5", 1),
        ("A5", 1),
        ("A5", 1),
        ("G5", 2),
        ("F5", 1),
        ("F5", 1),
        ("E5", 1),
        ("E5", 1),
        ("D5", 1),
        ("D5", 1),
        ("C5", 2),
    ]
    events = []
    for pitch_name, duration in twinkle:
        p = m21pitch.Pitch(pitch_name)
        events.append((int(p.midi), float(duration)))
    return events


def _parse_duration_values(value):
    if not value.strip():
        return []
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _quantize_duration(duration, allowed_durations):
    return min(allowed_durations, key=lambda candidate: abs(candidate - duration))


def _parse_key_root(key_name):
    normalized = key_name.strip().upper().replace("♭", "B").replace("♯", "#")
    if normalized in PITCH_CLASS_MAP:
        return PITCH_CLASS_MAP[normalized]
    raise ValueError(f"Unsupported key root: {key_name}")


def _build_scale_pitch_classes(key_root, mode):
    base = _parse_key_root(key_root)
    intervals = MAJOR_SCALE_INTERVALS if mode == "major" else MINOR_SCALE_INTERVALS
    return {(base + interval) % 12 for interval in intervals}


def _snap_midi_to_scale(midi_value, scale_pitch_classes):
    if midi_value % 12 in scale_pitch_classes:
        return midi_value
    best = midi_value
    best_distance = 128
    for candidate in range(128):
        if candidate % 12 in scale_pitch_classes:
            distance = abs(candidate - midi_value)
            if distance < best_distance:
                best = candidate
                best_distance = distance
    return best


def load_note_events_from_folder(folder, cache_path, refresh_cache=False):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    all_files = sorted(
        p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    cache = _load_cache(cache_path)
    cached_files = cache["files"]
    active_keys = {str(p.resolve()) for p in all_files}

    # Remove stale entries for deleted files.
    for key in list(cached_files.keys()):
        if key not in active_keys:
            del cached_files[key]

    all_events = []
    reused = 0

    for path in all_files:
        key = str(path.resolve())
        mtime_ns = path.stat().st_mtime_ns
        cached_entry = cached_files.get(key)

        if (
            not refresh_cache
            and cached_entry is not None
            and cached_entry.get("mtime_ns") == mtime_ns
        ):
            events = [(int(m), float(d)) for m, d in cached_entry.get("events", [])]
            all_events.extend(events)
            reused += 1
            continue

        start = time.perf_counter()
        try:
            score = converter.parse(str(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        events = []
        for n in score.recurse().notes:
            if isinstance(n, note.Note):
                events.append((int(n.pitch.midi), float(n.duration.quarterLength)))

        elapsed = time.perf_counter() - start
        print(f"Parsed {path} in {elapsed:.2f}s ({len(events)} notes)")

        cached_files[key] = {
            "mtime_ns": mtime_ns,
            "events": [[m, d] for m, d in events],
        }
        all_events.extend(events)

    _save_cache(cache_path, cache)

    if all_files:
        print(
            f"Data scan complete: {len(all_files)} files, "
            f"{reused} loaded from cache, {len(all_events)} notes total."
        )

    return all_events


def _transpose_score_to_reference_key(score):
    try:
        analyzed_key = score.analyze("key")
    except Exception:
        return score

    target_tonic = m21pitch.Pitch("C") if analyzed_key.mode == "major" else m21pitch.Pitch("A")
    transposition = m21interval.Interval(analyzed_key.tonic, target_tonic)
    return score.transpose(transposition)


def _classify_chord_quality(ch):
    root = ch.root()
    if root is None:
        return None

    root_pc = int(root.pitchClass)
    pcs = {(int(p.pitchClass) - root_pc) % 12 for p in ch.pitches}
    if len(pcs) < 3:
        return None

    if {0, 4, 7, 11}.issubset(pcs):
        return "maj7"
    if {0, 4, 7, 10}.issubset(pcs):
        return "7"
    if {0, 3, 7, 10}.issubset(pcs):
        return "min7"
    if {0, 4, 7}.issubset(pcs):
        return "maj"
    if {0, 3, 7}.issubset(pcs):
        return "min"
    if {0, 3, 6}.issubset(pcs):
        return "dim"
    return None


def _extract_chord_event(ch):
    root = ch.root()
    quality = _classify_chord_quality(ch)
    if root is None or quality not in ALLOWED_CHORD_QUALITIES:
        return None
    duration = _quantize_duration(float(ch.duration.quarterLength), CHORD_DURATION_GRID)
    return (int(root.pitchClass), quality, duration)


def load_chord_events_from_folder(folder, cache_path, refresh_cache=False):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []

    all_files = sorted(
        p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    cache = _load_cache(cache_path)
    cached_files = cache["files"]
    active_keys = {str(p.resolve()) for p in all_files}
    for key in list(cached_files.keys()):
        if key not in active_keys:
            del cached_files[key]

    all_events = []
    reused = 0

    for path in all_files:
        key = str(path.resolve())
        mtime_ns = path.stat().st_mtime_ns
        cached_entry = cached_files.get(key)

        if (
            not refresh_cache
            and cached_entry is not None
            and cached_entry.get("mtime_ns") == mtime_ns
            and cached_entry.get("schema") == "chord_v2"
        ):
            events = [
                (int(root_pc), str(quality), float(duration))
                for root_pc, quality, duration in cached_entry.get("events", [])
            ]
            all_events.extend(events)
            reused += 1
            continue

        start = time.perf_counter()
        try:
            score = converter.parse(str(path))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        normalized_score = _transpose_score_to_reference_key(score)
        chordified = normalized_score.chordify()
        by_bar = {}
        for el in chordified.recurse().notes:
            if not isinstance(el, m21chord.Chord):
                continue
            event = _extract_chord_event(el)
            if event is None:
                continue
            bar_index = int(float(el.offset) // CHORD_BAR_LENGTH)
            current = by_bar.get(bar_index)
            if current is None or float(el.duration.quarterLength) > current[1]:
                root_pc, quality, _ = event
                by_bar[bar_index] = ((root_pc, quality, CHORD_BAR_LENGTH), float(el.duration.quarterLength))

        events = [by_bar[i][0] for i in sorted(by_bar.keys())]

        elapsed = time.perf_counter() - start
        print(f"Parsed chords {path} in {elapsed:.2f}s ({len(events)} events)")

        cached_files[key] = {
            "schema": "chord_v2",
            "mtime_ns": mtime_ns,
            "events": [[root_pc, quality, duration] for root_pc, quality, duration in events],
        }
        all_events.extend(events)

    _save_cache(cache_path, cache)

    if all_files:
        print(
            f"Chord data scan complete: {len(all_files)} files, "
            f"{reused} loaded from cache, {len(all_events)} chord events total."
        )

    return all_events


def create_training_chord_events():
    return [
        (0, "maj", 4.0),   # C
        (5, "maj", 4.0),   # F
        (7, "maj", 4.0),   # G
        (0, "maj", 4.0),   # C
        (9, "min", 4.0),   # Am
        (5, "maj", 4.0),   # F
        (7, "maj", 4.0),   # G
        (0, "maj", 4.0),   # C
    ]


def build_and_train_model(data_folder, laplace_alpha, cache_path, refresh_cache, task):
    if task == "chord":
        training_events = load_chord_events_from_folder(
            data_folder,
            cache_path=cache_path,
            refresh_cache=refresh_cache,
        )

        if training_events:
            print(f"Loaded {len(training_events)} chord events from '{data_folder}'.")
        else:
            print(
                f"No supported files found in '{data_folder}'. "
                "Using built-in chord training data."
            )
            training_events = create_training_chord_events()

        return ChordMarkovModel.train_from_events(
            training_events,
            laplace_alpha=laplace_alpha,
        )

    training_events = load_note_events_from_folder(
        data_folder,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )

    if training_events:
        print(f"Loaded {len(training_events)} notes from '{data_folder}'.")
    else:
        print(
            f"No supported files found in '{data_folder}'. "
            "Using built-in Twinkle training data."
        )
        training_events = create_training_data_events()

    return FactorizedMelodyModel.train_from_events(
        training_events,
        laplace_alpha=laplace_alpha,
    )


def visualize_melody(melody):
    print(melody)
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Markov Chain Melody")
    part = stream.Part()
    for midi_value, duration in melody:
        note_obj = note.Note(quarterLength=duration)
        note_obj.pitch.midi = int(midi_value)
        part.append(note_obj)
    score.append(part)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"generated_melody_{timestamp}.musicxml"
    xml_path = score.write("musicxml", fp=output_file)
    os.startfile(xml_path)
    print(f"Saved and opened: {xml_path}")


def _chord_token_to_pitches(root_pc, quality, octave):
    root_midi = int((octave + 1) * 12 + root_pc)
    if quality == "maj":
        intervals = [0, 4, 7]
    elif quality == "min":
        intervals = [0, 3, 7]
    elif quality == "dim":
        intervals = [0, 3, 6]
    elif quality == "7":
        intervals = [0, 4, 7, 10]
    elif quality == "maj7":
        intervals = [0, 4, 7, 11]
    elif quality == "min7":
        intervals = [0, 3, 7, 10]
    else:
        intervals = [0, 7]
    return [max(0, min(127, root_midi + interval)) for interval in intervals]


def _chord_token_to_pitch_classes(root_pc, quality):
    if quality == "maj":
        intervals = [0, 4, 7]
    elif quality == "min":
        intervals = [0, 3, 7]
    elif quality == "dim":
        intervals = [0, 3, 6]
    elif quality == "7":
        intervals = [0, 4, 7, 10]
    elif quality == "maj7":
        intervals = [0, 4, 7, 11]
    elif quality == "min7":
        intervals = [0, 3, 7, 10]
    else:
        intervals = [0, 7]
    return {(int(root_pc) + interval) % 12 for interval in intervals}


def _chord_token_in_scale(token, scale_pitch_classes):
    root_pc, quality, _ = token
    chord_pitch_classes = _chord_token_to_pitch_classes(root_pc, quality)
    return chord_pitch_classes.issubset(set(scale_pitch_classes))


def _pick_cadence_chord_state(
    chord_model,
    root_pc,
    quality_preferences,
    scale_pitch_classes=None,
):
    states = list(chord_model.chord_model.states)
    if scale_pitch_classes is not None:
        states = [s for s in states if _chord_token_in_scale(s, scale_pitch_classes)]
    if not states:
        return None

    quality_rank = {quality: idx for idx, quality in enumerate(quality_preferences)}
    filtered = [s for s in states if int(s[0]) == int(root_pc) and str(s[1]) in quality_rank]
    if not filtered:
        filtered = [s for s in states if int(s[0]) == int(root_pc)]
    if not filtered:
        return None

    filtered.sort(
        key=lambda s: (
            quality_rank.get(str(s[1]), 999),
            abs(float(s[2]) - CHORD_BAR_LENGTH),
        )
    )
    best = filtered[0]
    return (int(best[0]), str(best[1]), float(CHORD_BAR_LENGTH))


def _apply_cadence_constraints(
    progression,
    chord_model,
    key_root_pc,
    mode,
    cadence_every_bars=4,
    scale_pitch_classes=None,
):
    if cadence_every_bars is None or int(cadence_every_bars) <= 1:
        return progression
    if key_root_pc is None:
        return progression
    if len(progression) < 2:
        return progression

    cadence_every_bars = int(cadence_every_bars)
    tonic_root = int(key_root_pc) % 12
    dominant_root = (tonic_root + 7) % 12

    if str(mode).lower() == "minor":
        tonic_quality_preferences = ["min", "min7"]
        dominant_quality_preferences = ["min", "min7", "7", "maj"]
    else:
        tonic_quality_preferences = ["maj", "maj7"]
        dominant_quality_preferences = ["7", "maj"]

    tonic_state = _pick_cadence_chord_state(
        chord_model,
        tonic_root,
        tonic_quality_preferences,
        scale_pitch_classes=scale_pitch_classes,
    )
    dominant_state = _pick_cadence_chord_state(
        chord_model,
        dominant_root,
        dominant_quality_preferences,
        scale_pitch_classes=scale_pitch_classes,
    )
    if tonic_state is None or dominant_state is None:
        return progression

    constrained = list(progression)
    cadence_ends = set(range(cadence_every_bars - 1, len(constrained), cadence_every_bars))
    cadence_ends.add(len(constrained) - 1)
    for end_idx in sorted(cadence_ends):
        if end_idx < 1:
            continue
        constrained[end_idx - 1] = dominant_state
        constrained[end_idx] = tonic_state

    return constrained


def _chord_token_to_roman_symbol(root_pc, quality, mode):
    mode = str(mode).lower()
    root_pc = int(root_pc) % 12
    quality = str(quality)

    if mode == "minor":
        interval_to_roman = {0: "i", 2: "ii", 3: "iii", 5: "iv", 7: "v", 8: "vi", 10: "vii"}
    else:
        interval_to_roman = {0: "i", 2: "ii", 4: "iii", 5: "iv", 7: "v", 9: "vi", 11: "vii"}

    base = interval_to_roman.get(root_pc)
    if base is None:
        return None

    if quality == "dim":
        return f"{base}°"
    if quality in {"maj", "maj7", "7"}:
        return base.upper()
    if quality in {"min", "min7"}:
        return base.lower()
    return None


def _extract_progression_symbols_from_filename(path):
    tokens = [t for t in Path(path).stem.split("_") if t]
    if not tokens:
        return None, []

    mode_idx = None
    mode = None
    for i, token in enumerate(tokens):
        lowered = token.strip().lower()
        if lowered in {"major", "minor"}:
            mode_idx = i
            mode = lowered
            break

    if mode_idx is None:
        return None, []

    end_idx = len(tokens)
    for i in range(mode_idx + 1, len(tokens)):
        if tokens[i].strip().lower() == "progression":
            end_idx = i
            break

    symbols = [s for s in tokens[mode_idx + 1 : end_idx] if s]
    return mode, symbols


def load_progression_blocks_from_folder(folder, cache_path, refresh_cache=False):
    folder_path = Path(folder)
    if not folder_path.exists():
        return {"major": [], "minor": []}

    all_files = sorted(
        p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    cache = _load_cache(cache_path)
    cached_files = cache["files"]
    active_keys = {str(p.resolve()) for p in all_files}
    for key in list(cached_files.keys()):
        if key not in active_keys:
            del cached_files[key]

    blocks_by_mode = {"major": [], "minor": []}
    reused = 0

    for path in all_files:
        key = str(path.resolve())
        mtime_ns = path.stat().st_mtime_ns
        cached_entry = cached_files.get(key)

        if (
            not refresh_cache
            and cached_entry is not None
            and cached_entry.get("mtime_ns") == mtime_ns
            and cached_entry.get("schema") == "progression_blocks_v1"
        ):
            mode = str(cached_entry.get("mode", ""))
            blocks = [tuple(block) for block in cached_entry.get("blocks", [])]
            if mode in blocks_by_mode:
                blocks_by_mode[mode].extend(blocks)
                reused += 1
            continue

        mode, symbols = _extract_progression_symbols_from_filename(path)
        blocks = []
        if mode in blocks_by_mode and symbols:
            for i in range(0, len(symbols) - 3, 4):
                block = tuple(symbols[i : i + 4])
                if len(block) == 4:
                    blocks.append(block)
                    blocks_by_mode[mode].append(block)

            cached_files[key] = {
                "schema": "progression_blocks_v1",
                "mtime_ns": mtime_ns,
                "mode": mode,
                "blocks": [list(block) for block in blocks],
            }
            continue

        try:
            score = converter.parse(str(path))
            analyzed_key = score.analyze("key")
            detected_mode = str(analyzed_key.mode).lower()
            if detected_mode.startswith("major"):
                mode = "major"
            elif detected_mode.startswith("minor"):
                mode = "minor"
            else:
                mode = detected_mode
            if mode not in blocks_by_mode:
                continue
        except Exception:
            continue

        normalized_score = _transpose_score_to_reference_key(score)
        chordified = normalized_score.chordify()
        by_bar = {}
        for el in chordified.recurse().notes:
            if not isinstance(el, m21chord.Chord):
                continue
            event = _extract_chord_event(el)
            if event is None:
                continue
            bar_index = int(float(el.offset) // CHORD_BAR_LENGTH)
            current = by_bar.get(bar_index)
            if current is None or float(el.duration.quarterLength) > current[1]:
                by_bar[bar_index] = (event, float(el.duration.quarterLength))

        events = [by_bar[i][0] for i in sorted(by_bar.keys())]
        symbols = [
            _chord_token_to_roman_symbol(root_pc, quality, mode)
            for root_pc, quality, _ in events
        ]
        symbols = [s for s in symbols if s is not None]

        blocks = []
        for i in range(0, len(symbols) - 3, 4):
            block = tuple(symbols[i : i + 4])
            if len(block) == 4:
                blocks.append(block)
                blocks_by_mode[mode].append(block)

        cached_files[key] = {
            "schema": "progression_blocks_v1",
            "mtime_ns": mtime_ns,
            "mode": mode,
            "blocks": [list(block) for block in blocks],
        }

    _save_cache(cache_path, cache)
    if all_files:
        print(
            f"Progression block scan complete: {len(all_files)} files, "
            f"{reused} loaded from cache, "
            f"{len(blocks_by_mode['major'])} major blocks, {len(blocks_by_mode['minor'])} minor blocks."
        )
    return blocks_by_mode


def _build_progression_block_model(mode, laplace_alpha=1.0, progression_blocks=None):
    del mode  # Mode is resolved before this call via progression_blocks_by_mode.
    blocks = list(progression_blocks) if progression_blocks else []
    if not blocks:
        raise ValueError("No progression blocks available for progression-block model.")
    model = SecondOrderMarkov(blocks, laplace_alpha=laplace_alpha)

    # Seed a cyclic training sequence so the block model learns transitions between blocks.
    seed_sequence = []
    total_blocks = len(blocks)
    for start in range(total_blocks):
        for offset in range(total_blocks):
            seed_sequence.append(blocks[(start + offset) % total_blocks])
    seed_sequence.extend([blocks[-1], blocks[0], blocks[1], blocks[-1], blocks[0]])

    model.train(seed_sequence)
    return model


def _fallback_quality_for_root(root_pc, quality_preferences, scale_pitch_classes):
    for quality in quality_preferences:
        if quality not in ALLOWED_CHORD_QUALITIES:
            continue
        token = (int(root_pc), str(quality), float(CHORD_BAR_LENGTH))
        if scale_pitch_classes is None or _chord_token_in_scale(token, scale_pitch_classes):
            return str(quality)
    for quality in sorted(ALLOWED_CHORD_QUALITIES):
        token = (int(root_pc), str(quality), float(CHORD_BAR_LENGTH))
        if scale_pitch_classes is None or _chord_token_in_scale(token, scale_pitch_classes):
            return str(quality)
    return "maj"


def _roman_symbol_to_chord_token(symbol, chord_model, key_root_pc, mode, scale_pitch_classes=None):
    raw = str(symbol).strip()
    with_seventh = raw.endswith("7")
    if with_seventh:
        raw = raw[:-1]
    diminished = "°" in raw
    raw = raw.replace("°", "")

    roman = raw.lower()
    intervals = ROMAN_INTERVALS_MINOR if str(mode).lower() == "minor" else ROMAN_INTERVALS_MAJOR
    if roman not in intervals:
        roman = "i"
    root_pc = (int(key_root_pc) + intervals[roman]) % 12

    is_upper = raw[:1].isupper()
    if diminished:
        quality_preferences = ["dim"]
    elif with_seventh:
        if roman == "v" and is_upper:
            quality_preferences = ["7", "maj"]
        elif is_upper:
            quality_preferences = ["maj7", "maj", "7"]
        else:
            quality_preferences = ["min7", "min"]
    else:
        quality_preferences = ["maj", "maj7"] if is_upper else ["min", "min7"]

    observed = _pick_cadence_chord_state(
        chord_model,
        root_pc=root_pc,
        quality_preferences=quality_preferences,
        scale_pitch_classes=scale_pitch_classes,
    )
    if observed is not None:
        return (int(observed[0]), str(observed[1]), float(CHORD_BAR_LENGTH))

    fallback_quality = _fallback_quality_for_root(
        root_pc,
        quality_preferences=quality_preferences,
        scale_pitch_classes=scale_pitch_classes,
    )
    return (int(root_pc), str(fallback_quality), float(CHORD_BAR_LENGTH))


def _generate_progression_block_chords(
    chord_model,
    bars,
    key_root_pc,
    mode,
    scale_pitch_classes=None,
    laplace_alpha=1.0,
    progression_blocks=None,
):
    blocks_per_phrase = 4
    block_model = _build_progression_block_model(
        mode=mode,
        laplace_alpha=laplace_alpha,
        progression_blocks=progression_blocks,
    )
    total_blocks = max(1, int(np.ceil(float(bars) / float(blocks_per_phrase))))
    sampled_blocks = block_model.generate(total_blocks)

    progression = []
    for block in sampled_blocks:
        for symbol in block:
            progression.append(
                _roman_symbol_to_chord_token(
                    symbol,
                    chord_model=chord_model,
                    key_root_pc=key_root_pc,
                    mode=mode,
                    scale_pitch_classes=scale_pitch_classes,
                )
            )
            if len(progression) >= int(bars):
                return progression[: int(bars)]
    return progression[: int(bars)]


def _generate_song_chord_progression(
    chord_model,
    bars,
    scale_pitch_classes=None,
    chord_repeat_penalty=0.0,
    key_root_pc=None,
    mode="major",
    cadence_every_bars=4,
    use_progression_blocks=True,
    progression_laplace_alpha=1.0,
    progression_blocks_by_mode=None,
):
    if use_progression_blocks and key_root_pc is not None:
        mode_key = "minor" if str(mode).lower() == "minor" else "major"
        learned_blocks = None
        if progression_blocks_by_mode:
            learned_blocks = progression_blocks_by_mode.get(mode_key) or None
        if learned_blocks:
            chord_progression = _generate_progression_block_chords(
                chord_model=chord_model,
                bars=bars,
                key_root_pc=key_root_pc,
                mode=mode,
                scale_pitch_classes=scale_pitch_classes,
                laplace_alpha=progression_laplace_alpha,
                progression_blocks=learned_blocks,
            )
        else:
            print(
                f"No extracted {mode_key} progression blocks found; "
                "falling back to raw chord-chain generation."
            )
            chord_progression = chord_model.generate(
                bars,
                duration_values=[CHORD_BAR_LENGTH],
                repeat_penalty=chord_repeat_penalty,
                scale_pitch_classes=scale_pitch_classes,
            )
    else:
        chord_progression = chord_model.generate(
            bars,
            duration_values=[CHORD_BAR_LENGTH],
            repeat_penalty=chord_repeat_penalty,
            scale_pitch_classes=scale_pitch_classes,
        )

    chord_progression = _apply_cadence_constraints(
        chord_progression,
        chord_model=chord_model,
        key_root_pc=key_root_pc,
        mode=mode,
        cadence_every_bars=cadence_every_bars,
        scale_pitch_classes=scale_pitch_classes,
    )
    return chord_progression


def _snap_midi_to_pitch_classes(midi_value, pitch_classes):
    if midi_value % 12 in pitch_classes:
        return midi_value
    best = midi_value
    best_distance = 128
    for candidate in range(128):
        if candidate % 12 in pitch_classes:
            distance = abs(candidate - midi_value)
            if distance < best_distance:
                best = candidate
                best_distance = distance
    return best


def _collect_candidates_in_range(pitch_classes, min_midi, max_midi):
    return [
        midi
        for midi in range(int(min_midi), int(max_midi) + 1)
        if midi % 12 in pitch_classes
    ]


def _choose_voice_pitch(
    candidates,
    target_midi,
    prev_midi=None,
    max_jump=7,
    lower_bound=None,
    upper_bound=None,
):
    if not candidates:
        return int(max(0, min(127, target_midi)))

    filtered = candidates
    if lower_bound is not None:
        filtered = [m for m in filtered if m >= int(lower_bound)]
    if upper_bound is not None:
        filtered = [m for m in filtered if m <= int(upper_bound)]
    if not filtered:
        filtered = candidates

    if prev_midi is not None:
        local = [m for m in filtered if abs(m - int(prev_midi)) <= int(max_jump)]
        if local:
            filtered = local

    return int(min(filtered, key=lambda m: abs(m - int(target_midi))))


def _pitch_motion(a, b):
    if b > a:
        return 1
    if b < a:
        return -1
    return 0


def _parallel_perfect_type(prev_a, prev_b, curr_a, curr_b):
    prev_motion_a = _pitch_motion(prev_a, curr_a)
    prev_motion_b = _pitch_motion(prev_b, curr_b)
    if prev_motion_a == 0 or prev_motion_b == 0:
        return None
    if prev_motion_a != prev_motion_b:
        return None

    prev_interval = abs(prev_a - prev_b) % 12
    curr_interval = abs(curr_a - curr_b) % 12
    if prev_interval == 0 and curr_interval == 0:
        return "octave"
    if prev_interval == 7 and curr_interval == 7:
        return "fifth"
    return None


def _score_sonority_transition(
    prev,
    curr,
    targets,
    repeat_streaks=None,
    repeated_note_base_penalty=CHORALE_REPEAT_NOTE_BASE_PENALTY,
):
    score = 0.0

    for voice in VOICE_ORDER:
        target = int(targets[voice])
        pitch_value = int(curr[voice])
        score += 0.18 * abs(pitch_value - target)

        if prev is None:
            continue

        leap = abs(pitch_value - int(prev[voice]))
        jump_limit = int(VOICE_MAX_JUMP[voice])
        if leap > jump_limit:
            score += 1.3 * (leap - jump_limit)
        if leap > 12:
            score += 2.0 * (leap - 12)
        if voice != "bass" and leap > 5:
            score += 0.8 * (leap - 5)
        if pitch_value == int(prev[voice]):
            streak = 0 if repeat_streaks is None else int(repeat_streaks.get(voice, 0))
            score += float(repeated_note_base_penalty) + (
                CHORALE_REPEAT_NOTE_STREAK_PENALTY * streak
            )

    if prev is not None:
        for a, b in combinations(VOICE_ORDER, 2):
            perfect_type = _parallel_perfect_type(
                int(prev[a]), int(prev[b]), int(curr[a]), int(curr[b])
            )
            if perfect_type == "octave":
                score += 8.0
            elif perfect_type == "fifth":
                score += 6.5

        bass_motion = _pitch_motion(int(prev["bass"]), int(curr["bass"]))
        soprano_motion = _pitch_motion(int(prev["soprano"]), int(curr["soprano"]))
        if bass_motion != 0 and soprano_motion != 0:
            if bass_motion != soprano_motion:
                score -= 0.35
            else:
                score += 0.35
                curr_outer_interval = abs(int(curr["soprano"]) - int(curr["bass"])) % 12
                prev_outer_interval = abs(int(prev["soprano"]) - int(prev["bass"])) % 12
                if curr_outer_interval in {0, 7} and prev_outer_interval not in {0, 7}:
                    score += 1.4

    return float(score)


def _build_ranked_voice_candidates(
    pitch_classes,
    voice,
    target_midi,
    prev_midi,
    max_candidates_per_voice,
):
    low, high = VOICE_RANGES[voice]
    candidates = _collect_candidates_in_range(pitch_classes, low, high)
    if not candidates:
        fallback = _snap_midi_to_pitch_classes(int(target_midi), pitch_classes)
        fallback = max(low, min(high, fallback))
        candidates = [fallback]

    def rank_cost(midi_value):
        cost = abs(int(midi_value) - int(target_midi))
        if prev_midi is not None:
            leap = abs(int(midi_value) - int(prev_midi))
            jump_limit = int(VOICE_MAX_JUMP[voice])
            if leap > jump_limit:
                cost += 2 * (leap - jump_limit)
        return cost

    ranked = sorted(set(candidates), key=rank_cost)
    return ranked[: max(1, int(max_candidates_per_voice))]


def _enumerate_sonority_candidates(
    voice_candidates,
    prev_sonority,
    targets,
    top_sonorities_per_state,
    repeat_streaks=None,
    repeated_note_base_penalty=CHORALE_REPEAT_NOTE_BASE_PENALTY,
):
    sonority_scored = []

    def collect_with_limits(limit_tb=19, limit_ta=12, limit_as=12):
        for bass in voice_candidates["bass"]:
            tenor_pool = [
                p for p in voice_candidates["tenor"] if bass < p <= bass + int(limit_tb)
            ]
            if not tenor_pool:
                continue
            for tenor in tenor_pool:
                alto_pool = [
                    p
                    for p in voice_candidates["alto"]
                    if tenor < p <= tenor + int(limit_ta)
                ]
                if not alto_pool:
                    continue
                for alto in alto_pool:
                    soprano_pool = [
                        p
                        for p in voice_candidates["soprano"]
                        if alto < p <= alto + int(limit_as)
                    ]
                    if not soprano_pool:
                        continue
                    for soprano in soprano_pool:
                        sonority = {
                            "bass": int(bass),
                            "tenor": int(tenor),
                            "alto": int(alto),
                            "soprano": int(soprano),
                        }
                        transition_cost = _score_sonority_transition(
                            prev_sonority,
                            sonority,
                            targets,
                            repeat_streaks=repeat_streaks,
                            repeated_note_base_penalty=repeated_note_base_penalty,
                        )
                        sonority_scored.append((transition_cost, sonority))

    collect_with_limits()
    if not sonority_scored:
        collect_with_limits(limit_tb=24, limit_ta=16, limit_as=16)

    if not sonority_scored:
        for bass in voice_candidates["bass"]:
            for tenor in voice_candidates["tenor"]:
                for alto in voice_candidates["alto"]:
                    for soprano in voice_candidates["soprano"]:
                        if not (bass < tenor < alto < soprano):
                            continue
                        sonority = {
                            "bass": int(bass),
                            "tenor": int(tenor),
                            "alto": int(alto),
                            "soprano": int(soprano),
                        }
                        transition_cost = _score_sonority_transition(
                            prev_sonority,
                            sonority,
                            targets,
                            repeat_streaks=repeat_streaks,
                            repeated_note_base_penalty=repeated_note_base_penalty,
                        )
                        sonority_scored.append((transition_cost, sonority))

    sonority_scored.sort(key=lambda pair: pair[0])
    limit = max(1, int(top_sonorities_per_state))
    return [(sonority, transition_cost) for transition_cost, sonority in sonority_scored[:limit]]


def visualize_chord_progression(progression, chord_octave=4):
    print(progression)
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Markov Chain Chord Progression")
    part = stream.Part()
    for root_pc, quality, duration in progression:
        chord_obj = m21chord.Chord(
            _chord_token_to_pitches(int(root_pc), str(quality), chord_octave),
            quarterLength=float(duration),
        )
        part.append(chord_obj)
    score.append(part)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"generated_chords_{timestamp}.musicxml"
    xml_path = score.write("musicxml", fp=output_file)
    os.startfile(xml_path)
    print(f"Saved and opened: {xml_path}")


def generate_song(
    melody_model,
    chord_model,
    bars,
    beats_per_bar,
    seed_midi=None,
    pitch_min=48,
    pitch_max=84,
    max_jump=7,
    scale_pitch_classes=None,
    chord_repeat_penalty=0.0,
    key_root_pc=None,
    mode="major",
    cadence_every_bars=4,
    use_progression_blocks=True,
    progression_laplace_alpha=1.0,
    progression_blocks_by_mode=None,
):
    chord_progression = _generate_song_chord_progression(
        chord_model=chord_model,
        bars=bars,
        scale_pitch_classes=scale_pitch_classes,
        chord_repeat_penalty=chord_repeat_penalty,
        key_root_pc=key_root_pc,
        mode=mode,
        cadence_every_bars=cadence_every_bars,
        use_progression_blocks=use_progression_blocks,
        progression_laplace_alpha=progression_laplace_alpha,
        progression_blocks_by_mode=progression_blocks_by_mode,
    )

    total_notes = bars * beats_per_bar
    note_duration = CHORD_BAR_LENGTH / beats_per_bar
    interval_sequence = melody_model.interval_model.generate(max(1, total_notes - 1))

    first_root, first_quality, _ = chord_progression[0]
    first_chord_classes = _chord_token_to_pitch_classes(first_root, first_quality)

    if seed_midi is None:
        current_midi = int(np.random.choice(melody_model.seed_pitches))
    else:
        current_midi = int(seed_midi)
    current_midi = max(pitch_min, min(pitch_max, current_midi))
    current_midi = _snap_midi_to_pitch_classes(current_midi, first_chord_classes)
    current_midi = max(pitch_min, min(pitch_max, current_midi))

    melody = []
    strong_beats = {0, beats_per_bar // 2}

    for idx in range(total_notes):
        bar_idx = idx // beats_per_bar
        beat_idx = idx % beats_per_bar
        root_pc, quality, _ = chord_progression[bar_idx]
        chord_classes = _chord_token_to_pitch_classes(root_pc, quality)

        if idx > 0:
            delta = int(interval_sequence[idx - 1])
            delta = max(-max_jump, min(max_jump, delta))
            current_midi += delta
            current_midi = max(pitch_min, min(pitch_max, current_midi))

        if beat_idx in strong_beats:
            current_midi = _snap_midi_to_pitch_classes(current_midi, chord_classes)
        elif scale_pitch_classes is not None:
            current_midi = _snap_midi_to_pitch_classes(current_midi, scale_pitch_classes)

        current_midi = max(pitch_min, min(pitch_max, current_midi))
        melody.append((int(current_midi), float(note_duration)))

    return chord_progression, melody


def generate_song_chorale(
    melody_model,
    chord_model,
    bars,
    beats_per_bar,
    scale_pitch_classes=None,
    chord_repeat_penalty=0.0,
    beam_width=20,
    max_candidates_per_voice=6,
    top_sonorities_per_state=24,
    repeated_note_base_penalty=CHORALE_REPEAT_NOTE_BASE_PENALTY,
    key_root_pc=None,
    mode="major",
    cadence_every_bars=4,
    use_progression_blocks=True,
    progression_laplace_alpha=1.0,
    progression_blocks_by_mode=None,
):
    chord_progression = _generate_song_chord_progression(
        chord_model=chord_model,
        bars=bars,
        scale_pitch_classes=scale_pitch_classes,
        chord_repeat_penalty=chord_repeat_penalty,
        key_root_pc=key_root_pc,
        mode=mode,
        cadence_every_bars=cadence_every_bars,
        use_progression_blocks=use_progression_blocks,
        progression_laplace_alpha=progression_laplace_alpha,
        progression_blocks_by_mode=progression_blocks_by_mode,
    )

    total_notes = bars * beats_per_bar
    note_duration = CHORD_BAR_LENGTH / beats_per_bar
    interval_sequences = {
        voice: melody_model.interval_model.generate(max(1, total_notes - 1))
        for voice in VOICE_ORDER
    }
    centers = {voice: (low + high) // 2 for voice, (low, high) in VOICE_RANGES.items()}
    beams = [
        {
            "score": 0.0,
            "last": None,
            "targets": dict(centers),
            "voices": {voice: [] for voice in VOICE_ORDER},
            "repeat_streaks": {voice: 0 for voice in VOICE_ORDER},
        }
    ]

    for idx in range(total_notes):
        bar_idx = idx // beats_per_bar
        root_pc, quality, _ = chord_progression[bar_idx]
        # Chorale mode stays chord-tone focused at every slot.
        allowed_classes = _chord_token_to_pitch_classes(root_pc, quality)

        expanded = []
        for beam in beams:
            prev_sonority = beam["last"]
            targets = dict(beam["targets"])

            if idx > 0:
                for voice in VOICE_ORDER:
                    delta = int(interval_sequences[voice][idx - 1])
                    delta = max(-VOICE_MAX_JUMP[voice], min(VOICE_MAX_JUMP[voice], delta))
                    targets[voice] = int(targets[voice] + delta)

            voice_candidates = {}
            for voice in VOICE_ORDER:
                prev_pitch = None if prev_sonority is None else int(prev_sonority[voice])
                voice_candidates[voice] = _build_ranked_voice_candidates(
                    pitch_classes=allowed_classes,
                    voice=voice,
                    target_midi=targets[voice],
                    prev_midi=prev_pitch,
                    max_candidates_per_voice=max_candidates_per_voice,
                )

            sonority_candidates = _enumerate_sonority_candidates(
                voice_candidates=voice_candidates,
                prev_sonority=prev_sonority,
                targets=targets,
                top_sonorities_per_state=top_sonorities_per_state,
                repeat_streaks=beam["repeat_streaks"],
                repeated_note_base_penalty=repeated_note_base_penalty,
            )

            for sonority, local_cost in sonority_candidates:
                next_repeat_streaks = {}
                for voice in VOICE_ORDER:
                    if prev_sonority is not None and int(sonority[voice]) == int(prev_sonority[voice]):
                        next_repeat_streaks[voice] = int(beam["repeat_streaks"][voice]) + 1
                    else:
                        next_repeat_streaks[voice] = 0
                next_voices = {
                    voice: beam["voices"][voice] + [(int(sonority[voice]), float(note_duration))]
                    for voice in VOICE_ORDER
                }
                expanded.append(
                    {
                        "score": float(beam["score"] + local_cost),
                        "last": sonority,
                        "targets": dict(sonority),
                        "voices": next_voices,
                        "repeat_streaks": next_repeat_streaks,
                    }
                )

        if not expanded:
            raise ValueError("Chorale generation failed to find SATB candidates.")

        expanded.sort(key=lambda item: item["score"])
        beams = expanded[: max(1, int(beam_width))]

    best = min(beams, key=lambda item: item["score"])
    return chord_progression, best["voices"]


def _default_chorale_beam():
    centers = {voice: (low + high) // 2 for voice, (low, high) in VOICE_RANGES.items()}
    return {
        "score": 0.0,
        "last": None,
        "targets": dict(centers),
        "repeat_streaks": {voice: 0 for voice in VOICE_ORDER},
    }


def _init_interval_pairs(melody_model):
    pairs = {}
    for voice in VOICE_ORDER:
        first, second = melody_model.interval_model._generate_starting_pair()
        pairs[voice] = (int(first), int(second))
    return pairs


def _consume_interval_delta(melody_model, interval_pairs, voice):
    prev_delta, curr_delta = interval_pairs[voice]
    next_delta = int(melody_model.interval_model._generate_next(prev_delta, curr_delta))
    interval_pairs[voice] = (curr_delta, next_delta)
    return int(curr_delta)


def _build_tension_settings(base_cadence_every, base_repeat_penalty, tension):
    tension = max(0.0, min(1.0, float(tension)))
    cadence_every = max(2, int(round(float(base_cadence_every) * (1.0 + 0.8 * tension))))
    repeat_penalty = max(0.0, float(base_repeat_penalty) * (1.0 - 0.6 * tension))
    return cadence_every, repeat_penalty


def _generate_realtime_chord_bar(
    bar_index,
    chord_model,
    progression_blocks_by_mode,
    key_root_pc,
    mode,
    scale_pitch_classes,
    chord_repeat_penalty,
    cadence_every_bars,
    use_progression_blocks,
    progression_laplace_alpha,
):
    mode_key = "minor" if str(mode).lower() == "minor" else "major"
    one_bar = _generate_song_chord_progression(
        chord_model=chord_model,
        bars=bar_index + 1,
        scale_pitch_classes=scale_pitch_classes,
        chord_repeat_penalty=chord_repeat_penalty,
        key_root_pc=key_root_pc,
        mode=mode,
        cadence_every_bars=cadence_every_bars,
        use_progression_blocks=use_progression_blocks,
        progression_laplace_alpha=progression_laplace_alpha,
        progression_blocks_by_mode=progression_blocks_by_mode,
    )
    if not one_bar:
        raise ValueError(f"Failed to generate chord bar for mode {mode_key}.")
    return one_bar[-1]


def _generate_realtime_chorale_bar(
    chord_token,
    melody_model,
    beats_per_bar,
    beam_width,
    max_candidates_per_voice,
    top_sonorities_per_state,
    repeated_note_base_penalty,
    interval_pairs,
    base_beam,
    global_note_index,
):
    allowed_classes = _chord_token_to_pitch_classes(int(chord_token[0]), str(chord_token[1]))
    beams = [dict(base_beam)]
    bar_notes = {voice: [] for voice in VOICE_ORDER}
    slots = int(beats_per_bar)
    note_duration = CHORD_BAR_LENGTH / max(1, int(beats_per_bar))

    for _ in range(slots):
        expanded = []
        step_deltas = {}
        if global_note_index > 0:
            for voice in VOICE_ORDER:
                step_deltas[voice] = _consume_interval_delta(melody_model, interval_pairs, voice)

        for beam in beams:
            prev_sonority = beam["last"]
            targets = dict(beam["targets"])

            if global_note_index > 0:
                for voice in VOICE_ORDER:
                    delta = int(step_deltas[voice])
                    delta = max(-VOICE_MAX_JUMP[voice], min(VOICE_MAX_JUMP[voice], delta))
                    targets[voice] = int(targets[voice] + delta)

            voice_candidates = {}
            for voice in VOICE_ORDER:
                prev_pitch = None if prev_sonority is None else int(prev_sonority[voice])
                voice_candidates[voice] = _build_ranked_voice_candidates(
                    pitch_classes=allowed_classes,
                    voice=voice,
                    target_midi=targets[voice],
                    prev_midi=prev_pitch,
                    max_candidates_per_voice=max_candidates_per_voice,
                )

            sonority_candidates = _enumerate_sonority_candidates(
                voice_candidates=voice_candidates,
                prev_sonority=prev_sonority,
                targets=targets,
                top_sonorities_per_state=top_sonorities_per_state,
                repeat_streaks=beam["repeat_streaks"],
                repeated_note_base_penalty=repeated_note_base_penalty,
            )

            for sonority, local_cost in sonority_candidates:
                next_repeat_streaks = {}
                for voice in VOICE_ORDER:
                    if prev_sonority is not None and int(sonority[voice]) == int(prev_sonority[voice]):
                        next_repeat_streaks[voice] = int(beam["repeat_streaks"][voice]) + 1
                    else:
                        next_repeat_streaks[voice] = 0

                expanded.append(
                    {
                        "score": float(beam["score"] + local_cost),
                        "last": sonority,
                        "targets": dict(sonority),
                        "repeat_streaks": next_repeat_streaks,
                    }
                )

        if not expanded:
            raise ValueError("Realtime chorale generation failed to find SATB candidates.")

        expanded.sort(key=lambda item: item["score"])
        beams = expanded[: max(1, int(beam_width))]
        committed = beams[0]["last"]
        for voice in VOICE_ORDER:
            bar_notes[voice].append((int(committed[voice]), float(note_duration)))
        global_note_index += 1

    return bar_notes, beams[0], global_note_index


def load_melody_slots_from_file(melody_input_path, beats_per_bar):
    score = converter.parse(str(melody_input_path))
    parts = score.parts
    # Use flattened offsets so note timing is absolute across all measures.
    source = parts[0].flatten() if len(parts) > 0 else score.flatten()

    step = CHORD_BAR_LENGTH / max(1, int(beats_per_bar))
    notes_data = []
    max_end = 0.0
    for el in source.notes:
        if not isinstance(el, note.Note):
            continue
        start = float(el.offset)
        end = start + float(el.duration.quarterLength)
        notes_data.append((start, end, int(el.pitch.midi)))
        if end > max_end:
            max_end = end

    if not notes_data:
        raise ValueError(f"No notes found in melody input: {melody_input_path}")

    notes_data.sort(key=lambda x: x[0])

    total_slots = max(1, int(np.ceil(max_end / step)))
    slots = []
    for i in range(total_slots):
        t = i * step
        chosen = None
        for start, end, midi_value in notes_data:
            if start <= t < end:
                chosen = midi_value
                break
        slots.append(chosen)

    return slots


def resolve_melody_input_path(melody_input, melody_input_folder):
    if melody_input is None:
        folder_path = Path(melody_input_folder)
        candidates = sorted(
            p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(
                f"No melody file found in '{folder_path}'. "
                "Add a MIDI/MusicXML file or pass --melody-input explicitly."
            )
        return candidates[0]

    raw_path = Path(melody_input)
    if raw_path.exists():
        return raw_path

    folder_path = Path(melody_input_folder)
    candidate = folder_path / raw_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Melody input not found: '{melody_input}'. "
        f"Tried '{raw_path}' and '{candidate}'."
    )


def _fit_chords_to_melody(
    chord_progression,
    melody_slots,
    beats_per_bar,
    chord_model,
    scale_pitch_classes=None,
):
    if not melody_slots:
        return chord_progression

    states = list(chord_model.chord_model.states)
    if scale_pitch_classes is not None:
        states = [s for s in states if _chord_token_in_scale(s, scale_pitch_classes)]
    if not states:
        return chord_progression

    fitted = []
    beats_per_bar = max(1, int(beats_per_bar))
    strong_slots = {0, beats_per_bar // 2}
    for bar_idx, original in enumerate(chord_progression):
        start = bar_idx * beats_per_bar
        end = start + beats_per_bar
        bar_slots = melody_slots[start:end]
        if not bar_slots:
            fitted.append(original)
            continue

        strong_pcs = [
            int(midi_value) % 12
            for i, midi_value in enumerate(bar_slots)
            if midi_value is not None and i in strong_slots
        ]
        weak_pcs = [
            int(midi_value) % 12
            for i, midi_value in enumerate(bar_slots)
            if midi_value is not None and i not in strong_slots
        ]
        if not strong_pcs and not weak_pcs:
            fitted.append(original)
            continue

        best_state = None
        best_score = -10**9
        for candidate in states:
            pcs = _chord_token_to_pitch_classes(int(candidate[0]), str(candidate[1]))
            score = 0
            score += 3 * sum(1 for pc in strong_pcs if pc in pcs)
            score += 1 * sum(1 for pc in weak_pcs if pc in pcs)
            score -= 2 * sum(1 for pc in strong_pcs if pc not in pcs)
            if (int(candidate[0]), str(candidate[1])) == (int(original[0]), str(original[1])):
                score += 1
            if score > best_score:
                best_score = score
                best_state = candidate

        if best_state is None:
            fitted.append(original)
        else:
            fitted.append((int(best_state[0]), str(best_state[1]), float(CHORD_BAR_LENGTH)))

    return fitted


def generate_harmonized_chorale(
    melody_model,
    chord_model,
    melody_slots,
    beats_per_bar,
    scale_pitch_classes=None,
    chord_repeat_penalty=0.0,
    beam_width=20,
    max_candidates_per_voice=6,
    top_sonorities_per_state=24,
    repeated_note_base_penalty=CHORALE_REPEAT_NOTE_BASE_PENALTY,
    key_root_pc=None,
    mode="major",
    cadence_every_bars=4,
    use_progression_blocks=True,
    progression_laplace_alpha=1.0,
    progression_blocks_by_mode=None,
):
    beats_per_bar = max(1, int(beats_per_bar))
    total_notes = len(melody_slots)
    bars = max(1, int(np.ceil(total_notes / beats_per_bar)))
    padded_slots = list(melody_slots) + [None] * (bars * beats_per_bar - total_notes)

    chord_progression = _generate_song_chord_progression(
        chord_model=chord_model,
        bars=bars,
        scale_pitch_classes=scale_pitch_classes,
        chord_repeat_penalty=chord_repeat_penalty,
        key_root_pc=key_root_pc,
        mode=mode,
        cadence_every_bars=cadence_every_bars,
        use_progression_blocks=use_progression_blocks,
        progression_laplace_alpha=progression_laplace_alpha,
        progression_blocks_by_mode=progression_blocks_by_mode,
    )
    chord_progression = _fit_chords_to_melody(
        chord_progression=chord_progression,
        melody_slots=padded_slots,
        beats_per_bar=beats_per_bar,
        chord_model=chord_model,
        scale_pitch_classes=scale_pitch_classes,
    )

    note_duration = CHORD_BAR_LENGTH / beats_per_bar
    interval_sequences = {
        voice: melody_model.interval_model.generate(max(1, total_notes - 1))
        for voice in VOICE_ORDER
    }
    centers = {voice: (low + high) // 2 for voice, (low, high) in VOICE_RANGES.items()}
    beams = [
        {
            "score": 0.0,
            "last": None,
            "targets": dict(centers),
            "voices": {voice: [] for voice in VOICE_ORDER},
            "repeat_streaks": {voice: 0 for voice in VOICE_ORDER},
        }
    ]

    for idx in range(total_notes):
        bar_idx = idx // beats_per_bar
        root_pc, quality, _ = chord_progression[bar_idx]
        allowed_classes = _chord_token_to_pitch_classes(root_pc, quality)
        fixed_soprano = padded_slots[idx]
        if fixed_soprano is not None:
            s_low, s_high = VOICE_RANGES["soprano"]
            fixed_soprano = int(max(s_low, min(s_high, int(fixed_soprano))))

        expanded = []
        for beam in beams:
            prev_sonority = beam["last"]
            targets = dict(beam["targets"])

            if idx > 0:
                for voice in VOICE_ORDER:
                    delta = int(interval_sequences[voice][idx - 1])
                    delta = max(-VOICE_MAX_JUMP[voice], min(VOICE_MAX_JUMP[voice], delta))
                    targets[voice] = int(targets[voice] + delta)
            if fixed_soprano is not None:
                targets["soprano"] = int(fixed_soprano)

            voice_candidates = {}
            for voice in VOICE_ORDER:
                prev_pitch = None if prev_sonority is None else int(prev_sonority[voice])
                if voice == "soprano" and fixed_soprano is not None:
                    voice_candidates[voice] = [int(fixed_soprano)]
                else:
                    voice_candidates[voice] = _build_ranked_voice_candidates(
                        pitch_classes=allowed_classes,
                        voice=voice,
                        target_midi=targets[voice],
                        prev_midi=prev_pitch,
                        max_candidates_per_voice=max_candidates_per_voice,
                    )

            sonority_candidates = _enumerate_sonority_candidates(
                voice_candidates=voice_candidates,
                prev_sonority=prev_sonority,
                targets=targets,
                top_sonorities_per_state=top_sonorities_per_state,
                repeat_streaks=beam["repeat_streaks"],
                repeated_note_base_penalty=repeated_note_base_penalty,
            )

            for sonority, local_cost in sonority_candidates:
                next_repeat_streaks = {}
                for voice in VOICE_ORDER:
                    if prev_sonority is not None and int(sonority[voice]) == int(prev_sonority[voice]):
                        next_repeat_streaks[voice] = int(beam["repeat_streaks"][voice]) + 1
                    else:
                        next_repeat_streaks[voice] = 0
                next_voices = {
                    voice: beam["voices"][voice] + [(int(sonority[voice]), float(note_duration))]
                    for voice in VOICE_ORDER
                }
                expanded.append(
                    {
                        "score": float(beam["score"] + local_cost),
                        "last": sonority,
                        "targets": dict(sonority),
                        "voices": next_voices,
                        "repeat_streaks": next_repeat_streaks,
                    }
                )

        if not expanded:
            raise ValueError("Harmonization failed to find SATB candidates.")

        expanded.sort(key=lambda item: item["score"])
        beams = expanded[: max(1, int(beam_width))]

    best = min(beams, key=lambda item: item["score"])
    return chord_progression, best["voices"]


def _parse_realtime_command(raw):
    text = raw.strip()
    if not text:
        return ("next", None)
    lowered = text.lower()
    if lowered in {"q", "quit", "exit", "stop"}:
        return ("stop", None)
    if lowered in {"h", "help", "?"}:
        return ("help", None)
    if lowered == "show":
        return ("show", None)

    match = re.match(r"^([a-zA-Z_]+)\s*=\s*(.+)$", text)
    if not match:
        parts = text.split(maxsplit=1)
        if len(parts) == 2:
            return (parts[0].lower(), parts[1].strip())
        return ("unknown", text)
    return (match.group(1).lower(), match.group(2).strip())


def run_realtime_session(args, melody_model, chord_model):
    progression_blocks_cache = "cache/progression_blocks_cache.json"
    progression_blocks_by_mode = load_progression_blocks_from_folder(
        args.chord_data_folder,
        cache_path=progression_blocks_cache,
        refresh_cache=args.refresh_cache,
    )

    state = {
        "key": str(args.key),
        "mode": str(args.mode),
        "beats_per_bar": int(args.beats_per_bar),
        "cadence_every_bars": int(args.cadence_every_bars),
        "tension": 0.0,
        "bar_index": 0,
        "global_note_index": 0,
        "beam": _default_chorale_beam(),
        "interval_pairs": _init_interval_pairs(melody_model),
        "voices": {voice: [] for voice in VOICE_ORDER},
        "chords": [],
    }

    def print_help():
        print("Commands: enter=next | key <C/G/F#...> | mode <major/minor> | density <1-8>")
        print("          cadence <N> | tension <0..1> | show | stop")

    print_help()
    max_bars = max(1, int(args.realtime_bars))
    live_snapshot_path = Path(args.realtime_live_file)
    opened_live_snapshot = False

    while state["bar_index"] < max_bars:
        try:
            scale_pitch_classes = None
            if not args.disable_scale_snap:
                scale_pitch_classes = _build_scale_pitch_classes(state["key"], state["mode"])
            key_root_pc = _parse_key_root(state["key"])
        except Exception as exc:
            print(f"Realtime config error: {exc}")
            break

        cadence_every_eff, repeat_penalty_eff = _build_tension_settings(
            state["cadence_every_bars"],
            args.chorale_repeat_note_penalty,
            state["tension"],
        )

        chord_bar = _generate_realtime_chord_bar(
            bar_index=state["bar_index"],
            chord_model=chord_model,
            progression_blocks_by_mode=progression_blocks_by_mode,
            key_root_pc=key_root_pc,
            mode=state["mode"],
            scale_pitch_classes=scale_pitch_classes,
            chord_repeat_penalty=args.chord_repeat_penalty,
            cadence_every_bars=cadence_every_eff,
            use_progression_blocks=not args.disable_progression_blocks,
            progression_laplace_alpha=args.progression_laplace_alpha,
        )

        bar_notes, next_beam, next_note_index = _generate_realtime_chorale_bar(
            chord_token=chord_bar,
            melody_model=melody_model,
            beats_per_bar=state["beats_per_bar"],
            beam_width=args.chorale_beam_width,
            max_candidates_per_voice=args.chorale_candidates_per_voice,
            top_sonorities_per_state=args.chorale_top_sonorities,
            repeated_note_base_penalty=repeat_penalty_eff,
            interval_pairs=state["interval_pairs"],
            base_beam=state["beam"],
            global_note_index=state["global_note_index"],
        )

        for voice in VOICE_ORDER:
            state["voices"][voice].extend(bar_notes[voice])
        state["chords"].append((int(chord_bar[0]), str(chord_bar[1]), float(CHORD_BAR_LENGTH)))
        state["beam"] = next_beam
        state["global_note_index"] = next_note_index
        state["bar_index"] += 1

        print(
            f"Bar {state['bar_index']:02d}: chord=({chord_bar[0]}, {chord_bar[1]}) "
            f"key={state['key']} {state['mode']} density={state['beats_per_bar']} "
            f"cadence={state['cadence_every_bars']} tension={state['tension']:.2f}"
        )
        if not args.disable_realtime_live_update:
            open_now = (not opened_live_snapshot) or bool(args.realtime_open_each_bar)
            _write_chorale_snapshot(state["voices"], live_snapshot_path, open_file=open_now)
            opened_live_snapshot = True

        if state["bar_index"] >= max_bars:
            break

        cmd = input("realtime> ").strip()
        action, value = _parse_realtime_command(cmd)

        if action == "next":
            continue
        if action == "stop":
            break
        if action == "help":
            print_help()
            continue
        if action == "show":
            print(
                f"state: key={state['key']} mode={state['mode']} density={state['beats_per_bar']} "
                f"cadence={state['cadence_every_bars']} tension={state['tension']:.2f}"
            )
            continue
        try:
            if action == "key":
                _parse_key_root(value)
                state["key"] = value
            elif action == "mode":
                if value.lower() not in {"major", "minor"}:
                    raise ValueError("mode must be major or minor")
                state["mode"] = value.lower()
            elif action in {"density", "beats", "beats_per_bar"}:
                beats = int(value)
                if beats < 1 or beats > 8:
                    raise ValueError("density must be between 1 and 8")
                state["beats_per_bar"] = beats
            elif action == "cadence":
                cadence = int(value)
                if cadence < 1:
                    raise ValueError("cadence must be >= 1")
                state["cadence_every_bars"] = cadence
            elif action == "tension":
                tension = float(value)
                state["tension"] = max(0.0, min(1.0, tension))
            else:
                print(f"Unknown command: {cmd}")
        except Exception as exc:
            print(f"Invalid command '{cmd}': {exc}")

    visualize_song_chorale(state["voices"])
    print("Realtime session finished.")


def visualize_song(melody, chord_progression, chord_octave=4):
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Markov Song (Melody + Chords)")

    melody_part = stream.Part()
    melody_part.id = "Melody"
    for midi_value, duration in melody:
        n = note.Note(quarterLength=float(duration))
        n.pitch.midi = int(midi_value)
        melody_part.append(n)

    chord_part = stream.Part()
    chord_part.id = "Chords"
    for root_pc, quality, duration in chord_progression:
        c = m21chord.Chord(
            _chord_token_to_pitches(int(root_pc), str(quality), chord_octave),
            quarterLength=float(duration),
        )
        chord_part.append(c)

    score.append(melody_part)
    score.append(chord_part)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"generated_song_{timestamp}.musicxml"
    xml_path = score.write("musicxml", fp=output_file)
    os.startfile(xml_path)
    print(f"Saved and opened: {xml_path}")


def visualize_song_chorale(voices):
    score = _build_chorale_score(voices)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_file = f"generated_song_{timestamp}.musicxml"
    xml_path = score.write("musicxml", fp=output_file)
    os.startfile(xml_path)
    print(f"Saved and opened: {xml_path}")


def _build_chorale_score(voices):
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Markov Song (SATB Chorale)")

    for voice_name in ["soprano", "alto", "tenor", "bass"]:
        part = stream.Part()
        part.id = voice_name.capitalize()
        for midi_value, duration in voices[voice_name]:
            n = note.Note(quarterLength=float(duration))
            n.pitch.midi = int(midi_value)
            part.append(n)
        score.append(part)

    return score


def _write_chorale_snapshot(voices, output_file, open_file=False):
    score = _build_chorale_score(voices)
    xml_path = score.write("musicxml", fp=str(output_file))
    if open_file:
        os.startfile(xml_path)
    return xml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Markov generator for melody or chord progression"
    )
    parser.add_argument(
        "--task",
        choices=["melody", "chord", "song", "realtime", "harmonize"],
        default="song",
        help="Select generation task.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain from data and overwrite saved model.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to saved model file. Defaults by task.",
    )
    parser.add_argument(
        "--data-folder",
        default=None,
        help="Folder containing MIDI/MusicXML files for training. Defaults by task.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=40,
        help="Number of notes to generate.",
    )
    parser.add_argument(
        "--laplace-alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing alpha used during training.",
    )
    parser.add_argument(
        "--seed-midi",
        type=int,
        default=None,
        help="Optional starting MIDI pitch (0-127).",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Path to cache file for faster retraining. Defaults by task.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore parse cache and rebuild it from source files.",
    )
    parser.add_argument(
        "--pitch-min",
        type=int,
        default=48,
        help="Lowest allowed MIDI pitch during generation.",
    )
    parser.add_argument(
        "--pitch-max",
        type=int,
        default=84,
        help="Highest allowed MIDI pitch during generation.",
    )
    parser.add_argument(
        "--max-jump",
        type=int,
        default=7,
        help="Maximum semitone jump allowed between consecutive notes.",
    )
    parser.add_argument(
        "--key",
        default="C",
        help="Key root for scale snapping (e.g. C, G, F#, Bb).",
    )
    parser.add_argument(
        "--mode",
        choices=["major", "minor"],
        default="major",
        help="Scale mode used when snapping to scale.",
    )
    parser.add_argument(
        "--disable-scale-snap",
        action="store_true",
        help="Disable scale snapping.",
    )
    parser.add_argument(
        "--duration-values",
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated duration values for quantization.",
    )
    parser.add_argument(
        "--disable-duration-quantize",
        action="store_true",
        help="Disable duration quantization.",
    )
    parser.add_argument(
        "--chord-octave",
        type=int,
        default=4,
        help="Octave used for rendering generated chords.",
    )
    parser.add_argument(
        "--chord-repeat-penalty",
        type=float,
        default=0.5,
        help="Penalty for repeating the same chord identity consecutively (0.0-0.99).",
    )
    parser.add_argument(
        "--song-bars",
        type=int,
        default=16,
        help="Number of bars for --task song.",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="Melody note slots per bar for --task song.",
    )
    parser.add_argument(
        "--song-style",
        choices=["lead", "chorale"],
        default="chorale",
        help="Song layout: lead melody + block chords, or SATB chorale.",
    )
    parser.add_argument(
        "--chorale-beam-width",
        type=int,
        default=20,
        help="Beam width for SATB chorale search (higher is slower, often better).",
    )
    parser.add_argument(
        "--chorale-candidates-per-voice",
        type=int,
        default=6,
        help="Per-voice candidate count before SATB combination in chorale mode.",
    )
    parser.add_argument(
        "--chorale-top-sonorities",
        type=int,
        default=24,
        help="Top local sonorities expanded per beam item in chorale mode.",
    )
    parser.add_argument(
        "--cadence-every-bars",
        type=int,
        default=4,
        help="Force V-I cadence every N bars in song modes (<=1 disables).",
    )
    parser.add_argument(
        "--disable-progression-blocks",
        action="store_true",
        help="Disable progression-block Markov transitions and use raw chord-chain generation.",
    )
    parser.add_argument(
        "--progression-laplace-alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing alpha for progression-block transition model.",
    )
    parser.add_argument(
        "--chorale-repeat-note-penalty",
        type=float,
        default=CHORALE_REPEAT_NOTE_BASE_PENALTY,
        help="Base penalty for repeating the same note in the same voice in chorale mode.",
    )
    parser.add_argument(
        "--melody-model-path",
        default="models/factorized_markov_model.npz",
        help="Melody model path for --task song.",
    )
    parser.add_argument(
        "--chord-model-path",
        default="models/chord_markov_model.npz",
        help="Chord model path for --task song.",
    )
    parser.add_argument(
        "--melody-data-folder",
        default="data",
        help="Melody training folder for --task song retraining.",
    )
    parser.add_argument(
        "--chord-data-folder",
        default="dataChords",
        help="Chord training folder for --task song retraining.",
    )
    parser.add_argument(
        "--realtime-bars",
        type=int,
        default=16,
        help="Number of bars to generate in realtime task.",
    )
    parser.add_argument(
        "--disable-realtime-live-update",
        action="store_true",
        help="Disable writing/updating a live MusicXML snapshot after each realtime bar.",
    )
    parser.add_argument(
        "--realtime-open-each-bar",
        action="store_true",
        help="Re-open the live MusicXML file on every bar update (can open many tabs/windows).",
    )
    parser.add_argument(
        "--realtime-live-file",
        default="generated_realtime_live.musicxml",
        help="Live MusicXML snapshot path updated each bar in realtime mode.",
    )
    parser.add_argument(
        "--melody-input",
        default=None,
        help="Input melody file path for --task harmonize (MIDI/MusicXML).",
    )
    parser.add_argument(
        "--melody-input-folder",
        default=DEFAULT_MELODY_INPUT_FOLDER,
        help="Default folder searched for --melody-input files when a direct path is not found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "realtime":
        if args.realtime_bars < 1:
            raise ValueError("--realtime-bars must be >= 1")
        if args.beats_per_bar < 1:
            raise ValueError("--beats-per-bar must be >= 1")
        if args.chorale_beam_width < 1:
            raise ValueError("--chorale-beam-width must be >= 1")
        if args.chorale_candidates_per_voice < 1:
            raise ValueError("--chorale-candidates-per-voice must be >= 1")
        if args.chorale_top_sonorities < 1:
            raise ValueError("--chorale-top-sonorities must be >= 1")
        if args.chorale_repeat_note_penalty < 0:
            raise ValueError("--chorale-repeat-note-penalty must be >= 0")
        if args.progression_laplace_alpha < 0:
            raise ValueError("--progression-laplace-alpha must be >= 0")

        melody_model_path = Path(args.melody_model_path)
        chord_model_path = Path(args.chord_model_path)
        melody_cache = "cache/note_events_cache.json"
        chord_cache = "cache/chord_events_cache.json"

        if args.retrain or not melody_model_path.exists():
            reason = "--retrain was set" if args.retrain else "melody model file not found"
            print(f"Training melody model because {reason}.")
            melody_model = build_and_train_model(
                args.melody_data_folder,
                args.laplace_alpha,
                cache_path=melody_cache,
                refresh_cache=args.refresh_cache,
                task="melody",
            )
            melody_model.save(melody_model_path)
            print(f"Saved trained melody model to '{melody_model_path}'.")
        else:
            melody_model = FactorizedMelodyModel.load(melody_model_path)
            print(f"Loaded existing melody model from '{melody_model_path}'.")

        if args.retrain or not chord_model_path.exists():
            reason = "--retrain was set" if args.retrain else "chord model file not found"
            print(f"Training chord model because {reason}.")
            chord_model = build_and_train_model(
                args.chord_data_folder,
                args.laplace_alpha,
                cache_path=chord_cache,
                refresh_cache=args.refresh_cache,
                task="chord",
            )
            chord_model.save(chord_model_path)
            print(f"Saved trained chord model to '{chord_model_path}'.")
        else:
            chord_model = ChordMarkovModel.load(chord_model_path)
            print(f"Loaded existing chord model from '{chord_model_path}'.")

        run_realtime_session(args, melody_model=melody_model, chord_model=chord_model)
        return

    if args.task == "harmonize":
        Path(args.melody_input_folder).mkdir(parents=True, exist_ok=True)
        if args.beats_per_bar < 1:
            raise ValueError("--beats-per-bar must be >= 1")
        if args.chorale_beam_width < 1:
            raise ValueError("--chorale-beam-width must be >= 1")
        if args.chorale_candidates_per_voice < 1:
            raise ValueError("--chorale-candidates-per-voice must be >= 1")
        if args.chorale_top_sonorities < 1:
            raise ValueError("--chorale-top-sonorities must be >= 1")
        if args.chorale_repeat_note_penalty < 0:
            raise ValueError("--chorale-repeat-note-penalty must be >= 0")
        if args.progression_laplace_alpha < 0:
            raise ValueError("--progression-laplace-alpha must be >= 0")

        melody_model_path = Path(args.melody_model_path)
        chord_model_path = Path(args.chord_model_path)
        melody_cache = "cache/note_events_cache.json"
        chord_cache = "cache/chord_events_cache.json"

        if args.retrain or not melody_model_path.exists():
            reason = "--retrain was set" if args.retrain else "melody model file not found"
            print(f"Training melody model because {reason}.")
            melody_model = build_and_train_model(
                args.melody_data_folder,
                args.laplace_alpha,
                cache_path=melody_cache,
                refresh_cache=args.refresh_cache,
                task="melody",
            )
            melody_model.save(melody_model_path)
            print(f"Saved trained melody model to '{melody_model_path}'.")
        else:
            melody_model = FactorizedMelodyModel.load(melody_model_path)
            print(f"Loaded existing melody model from '{melody_model_path}'.")

        if args.retrain or not chord_model_path.exists():
            reason = "--retrain was set" if args.retrain else "chord model file not found"
            print(f"Training chord model because {reason}.")
            chord_model = build_and_train_model(
                args.chord_data_folder,
                args.laplace_alpha,
                cache_path=chord_cache,
                refresh_cache=args.refresh_cache,
                task="chord",
            )
            chord_model.save(chord_model_path)
            print(f"Saved trained chord model to '{chord_model_path}'.")
        else:
            chord_model = ChordMarkovModel.load(chord_model_path)
            print(f"Loaded existing chord model from '{chord_model_path}'.")

        progression_blocks_cache = "cache/progression_blocks_cache.json"
        progression_blocks_by_mode = load_progression_blocks_from_folder(
            args.chord_data_folder,
            cache_path=progression_blocks_cache,
            refresh_cache=args.refresh_cache,
        )

        scale_pitch_classes = None
        if not args.disable_scale_snap:
            scale_pitch_classes = _build_scale_pitch_classes(args.key, args.mode)
        key_root_pc = _parse_key_root(args.key)
        melody_input_path = resolve_melody_input_path(
            args.melody_input,
            args.melody_input_folder,
        )
        print(f"Using melody input: {melody_input_path}")
        melody_slots = load_melody_slots_from_file(melody_input_path, args.beats_per_bar)

        _, voices = generate_harmonized_chorale(
            melody_model=melody_model,
            chord_model=chord_model,
            melody_slots=melody_slots,
            beats_per_bar=args.beats_per_bar,
            scale_pitch_classes=scale_pitch_classes,
            chord_repeat_penalty=args.chord_repeat_penalty,
            beam_width=args.chorale_beam_width,
            max_candidates_per_voice=args.chorale_candidates_per_voice,
            top_sonorities_per_state=args.chorale_top_sonorities,
            repeated_note_base_penalty=args.chorale_repeat_note_penalty,
            key_root_pc=key_root_pc,
            mode=args.mode,
            cadence_every_bars=args.cadence_every_bars,
            use_progression_blocks=not args.disable_progression_blocks,
            progression_laplace_alpha=args.progression_laplace_alpha,
            progression_blocks_by_mode=progression_blocks_by_mode,
        )
        visualize_song_chorale(voices)
        return

    if args.task == "song":
        if args.song_bars < 1:
            raise ValueError("--song-bars must be >= 1")
        if args.beats_per_bar < 1:
            raise ValueError("--beats-per-bar must be >= 1")
        if args.chorale_beam_width < 1:
            raise ValueError("--chorale-beam-width must be >= 1")
        if args.chorale_candidates_per_voice < 1:
            raise ValueError("--chorale-candidates-per-voice must be >= 1")
        if args.chorale_top_sonorities < 1:
            raise ValueError("--chorale-top-sonorities must be >= 1")
        if args.chorale_repeat_note_penalty < 0:
            raise ValueError("--chorale-repeat-note-penalty must be >= 0")
        if args.progression_laplace_alpha < 0:
            raise ValueError("--progression-laplace-alpha must be >= 0")

        melody_model_path = Path(args.melody_model_path)
        chord_model_path = Path(args.chord_model_path)
        melody_cache = "cache/note_events_cache.json"
        chord_cache = "cache/chord_events_cache.json"

        if args.retrain or not melody_model_path.exists():
            reason = "--retrain was set" if args.retrain else "melody model file not found"
            print(f"Training melody model because {reason}.")
            melody_model = build_and_train_model(
                args.melody_data_folder,
                args.laplace_alpha,
                cache_path=melody_cache,
                refresh_cache=args.refresh_cache,
                task="melody",
            )
            melody_model.save(melody_model_path)
            print(f"Saved trained melody model to '{melody_model_path}'.")
        else:
            melody_model = FactorizedMelodyModel.load(melody_model_path)
            print(f"Loaded existing melody model from '{melody_model_path}'.")

        if args.retrain or not chord_model_path.exists():
            reason = "--retrain was set" if args.retrain else "chord model file not found"
            print(f"Training chord model because {reason}.")
            chord_model = build_and_train_model(
                args.chord_data_folder,
                args.laplace_alpha,
                cache_path=chord_cache,
                refresh_cache=args.refresh_cache,
                task="chord",
            )
            chord_model.save(chord_model_path)
            print(f"Saved trained chord model to '{chord_model_path}'.")
        else:
            chord_model = ChordMarkovModel.load(chord_model_path)
            print(f"Loaded existing chord model from '{chord_model_path}'.")

        scale_pitch_classes = None
        if not args.disable_scale_snap:
            scale_pitch_classes = _build_scale_pitch_classes(args.key, args.mode)
        key_root_pc = _parse_key_root(args.key)
        progression_blocks_cache = "cache/progression_blocks_cache.json"
        progression_blocks_by_mode = load_progression_blocks_from_folder(
            args.chord_data_folder,
            cache_path=progression_blocks_cache,
            refresh_cache=args.refresh_cache,
        )

        if args.song_style == "chorale":
            _, voices = generate_song_chorale(
                melody_model=melody_model,
                chord_model=chord_model,
                bars=args.song_bars,
                beats_per_bar=args.beats_per_bar,
                scale_pitch_classes=scale_pitch_classes,
                chord_repeat_penalty=args.chord_repeat_penalty,
                beam_width=args.chorale_beam_width,
                max_candidates_per_voice=args.chorale_candidates_per_voice,
                top_sonorities_per_state=args.chorale_top_sonorities,
                repeated_note_base_penalty=args.chorale_repeat_note_penalty,
                key_root_pc=key_root_pc,
                mode=args.mode,
                cadence_every_bars=args.cadence_every_bars,
                use_progression_blocks=not args.disable_progression_blocks,
                progression_laplace_alpha=args.progression_laplace_alpha,
                progression_blocks_by_mode=progression_blocks_by_mode,
            )
            visualize_song_chorale(voices)
        else:
            chord_progression, melody = generate_song(
                melody_model=melody_model,
                chord_model=chord_model,
                bars=args.song_bars,
                beats_per_bar=args.beats_per_bar,
                seed_midi=args.seed_midi,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
                max_jump=args.max_jump,
                scale_pitch_classes=scale_pitch_classes,
                chord_repeat_penalty=args.chord_repeat_penalty,
                key_root_pc=key_root_pc,
                mode=args.mode,
                cadence_every_bars=args.cadence_every_bars,
                use_progression_blocks=not args.disable_progression_blocks,
                progression_laplace_alpha=args.progression_laplace_alpha,
                progression_blocks_by_mode=progression_blocks_by_mode,
            )
            visualize_song(melody, chord_progression, chord_octave=args.chord_octave)
        return

    if args.model_path is None:
        default_model_path = (
            "models/chord_markov_model.npz"
            if args.task == "chord"
            else "models/factorized_markov_model.npz"
        )
        model_path = Path(default_model_path)
    else:
        model_path = Path(args.model_path)

    default_cache_path = (
        "cache/chord_events_cache.json"
        if args.task == "chord"
        else "cache/note_events_cache.json"
    )
    cache_path = args.cache_path if args.cache_path else default_cache_path
    default_data_folder = "dataChords" if args.task == "chord" else "data"
    data_folder = args.data_folder if args.data_folder else default_data_folder

    if args.retrain or not model_path.exists():
        reason = "--retrain was set" if args.retrain else "model file not found"
        print(f"Training model because {reason}.")
        model = build_and_train_model(
            data_folder,
            args.laplace_alpha,
            cache_path=cache_path,
            refresh_cache=args.refresh_cache,
            task=args.task,
        )
        model.save(model_path)
        print(f"Saved trained model to '{model_path}'.")
    else:
        model = (
            ChordMarkovModel.load(model_path)
            if args.task == "chord"
            else FactorizedMelodyModel.load(model_path)
        )
        print(f"Loaded existing model from '{model_path}'.")

    if args.task == "chord":
        duration_values = None
        if not args.disable_duration_quantize:
            duration_values = _parse_duration_values(args.duration_values)
        scale_pitch_classes = None
        if not args.disable_scale_snap:
            scale_pitch_classes = _build_scale_pitch_classes(args.key, args.mode)
        progression = model.generate(
            args.length,
            duration_values=duration_values,
            repeat_penalty=args.chord_repeat_penalty,
            scale_pitch_classes=scale_pitch_classes,
        )
        visualize_chord_progression(progression, chord_octave=args.chord_octave)
    else:
        scale_pitch_classes = None
        if not args.disable_scale_snap:
            scale_pitch_classes = _build_scale_pitch_classes(args.key, args.mode)

        duration_values = None
        if not args.disable_duration_quantize:
            duration_values = _parse_duration_values(args.duration_values)

        generated_melody = model.generate_constrained(
            args.length,
            seed_midi=args.seed_midi,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            max_jump=args.max_jump,
            scale_pitch_classes=scale_pitch_classes,
            duration_values=duration_values,
        )
        visualize_melody(generated_melody)


if __name__ == "__main__":
    main()


