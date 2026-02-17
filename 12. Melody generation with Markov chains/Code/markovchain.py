import argparse
import json
import os
import time
from datetime import datetime
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

    def generate(self, length, duration_values=None, repeat_penalty=0.0):
        progression = self._generate_with_repeat_penalty(
            length, repeat_penalty=repeat_penalty
        )
        if duration_values:
            allowed = sorted(set(float(d) for d in duration_values))
            progression = [
                (root_pc, quality, _quantize_duration(duration, allowed))
                for root_pc, quality, duration in progression
            ]
        return progression

    def _generate_with_repeat_penalty(self, length, repeat_penalty=0.0):
        if length <= 0:
            return []
        if repeat_penalty <= 0:
            return self.chord_model.generate(length)

        repeat_penalty = max(0.0, min(0.99, float(repeat_penalty)))
        keep_factor = 1.0 - repeat_penalty

        if length == 1:
            first, _ = self.chord_model._generate_starting_pair()
            return [first]

        first, second = self.chord_model._generate_starting_pair()
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

            total = probs.sum()
            if total <= 0:
                probs = self.chord_model.transition_matrix[i, j]
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
):
    chord_progression = chord_model.generate(
        bars,
        duration_values=[CHORD_BAR_LENGTH],
        repeat_penalty=chord_repeat_penalty,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Markov generator for melody or chord progression"
    )
    parser.add_argument(
        "--task",
        choices=["melody", "chord", "song"],
        default="melody",
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
        default=8,
        help="Number of bars for --task song.",
    )
    parser.add_argument(
        "--beats-per-bar",
        type=int,
        default=4,
        help="Melody note slots per bar for --task song.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "song":
        if args.song_bars < 1:
            raise ValueError("--song-bars must be >= 1")
        if args.beats_per_bar < 1:
            raise ValueError("--beats-per-bar must be >= 1")

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
        progression = model.generate(
            args.length,
            duration_values=duration_values,
            repeat_penalty=args.chord_repeat_penalty,
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
