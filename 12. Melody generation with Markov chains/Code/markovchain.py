import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from music21 import converter, metadata, note, pitch as m21pitch, stream

SUPPORTED_EXTENSIONS = {".mid", ".midi", ".xml", ".musicxml", ".mxl"}
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


def build_and_train_model(data_folder, laplace_alpha, cache_path, refresh_cache):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Factorized Markov melody generator (intervals + durations)"
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain from data and overwrite saved model.",
    )
    parser.add_argument(
        "--model-path",
        default="models/factorized_markov_model.npz",
        help="Path to saved model file.",
    )
    parser.add_argument(
        "--data-folder",
        default="data",
        help="Folder containing MIDI/MusicXML files for training.",
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
        default="cache/note_events_cache.json",
        help="Path to parsed-note cache file for faster retraining.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path)

    if args.retrain or not model_path.exists():
        reason = "--retrain was set" if args.retrain else "model file not found"
        print(f"Training model because {reason}.")
        model = build_and_train_model(
            args.data_folder,
            args.laplace_alpha,
            cache_path=args.cache_path,
            refresh_cache=args.refresh_cache,
        )
        model.save(model_path)
        print(f"Saved trained model to '{model_path}'.")
    else:
        model = FactorizedMelodyModel.load(model_path)
        print(f"Loaded existing model from '{model_path}'.")

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
