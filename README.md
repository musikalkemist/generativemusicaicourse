# Markov Symbolic Music Generator

This repository generates symbolic music from MIDI/MusicXML using second-order Markov models.

Supported tasks:
- `melody`: melody only
- `chord`: chord progression only
- `song`: melody + chords together
- `realtime`: interactive bar-by-bar chorale generation
- `harmonize`: harmonize an input melody into SATB

Main script:
- `Code/markovchain.py`

Parameter reference:
- `Code/PARAMETERS.md`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `music21` once so score files open correctly:

```bash
python -m music21.configure
```

## Data

Place training files here:
- Melody data: `data/`
- Chord data: `dataChords/`

Supported formats:
- `.mid`, `.midi`, `.xml`, `.musicxml`, `.mxl`

## Quick Start

### Melody

Train:
```bash
python "Code/markovchain.py" --task melody --retrain
```

Generate from saved model:
```bash
python "Code/markovchain.py" --task melody
```

### Chords

Train:
```bash
python "Code/markovchain.py" --task chord --retrain --refresh-cache
```

Generate:
```bash
python "Code/markovchain.py" --task chord --length 16
```

### Song (Melody + Chords)

Train both models and generate:
```bash
python "Code/markovchain.py" --task song --retrain --refresh-cache
```

Generate from saved models:
```bash
python "Code/markovchain.py" --task song --song-bars 8 --beats-per-bar 4
```

Generate SATB chorale (4 voices conditioned on chord progression):
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 8 --beats-per-bar 4
```

Higher-quality chorale search (slower):
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 8 --beats-per-bar 4 --chorale-beam-width 24 --chorale-candidates-per-voice 7 --chorale-top-sonorities 30
```

Chorale with stronger repeat-note penalty + cadence forcing:
```bash
python "Code/markovchain.py" --task song --song-style chorale --song-bars 8 --beats-per-bar 4 --key A --mode major --chorale-repeat-note-penalty 6 --cadence-every-bars 4
```

Disable progression-block transitions and fall back to raw chord-chain generation:
```bash
python "Code/markovchain.py" --task song --song-style chorale --disable-progression-blocks
```

Progression blocks are now learned from `dataChords` (split into major/minor sets).
If a requested mode has no extracted blocks, song generation falls back to raw chord-chain generation.

### Realtime (Interactive)

```bash
python "Code/markovchain.py" --task realtime --realtime-bars 16
```

Realtime mode writes `generated_realtime_live.musicxml` after each bar.
Use `--realtime-open-each-bar` if your MuseScore setup does not auto-refresh file changes.

Live commands at each prompt:
- `key <C/G/F#...>`
- `mode <major/minor>`
- `density <1-8>`
- `cadence <N>`
- `tension <0..1>`
- `show`, `stop`

### Harmonize Input Melody

```bash
python "Code/markovchain.py" --task harmonize --melody-input "path/to/melody.musicxml" --key C --mode major
```

You can also place files in `melodyInput/` and pass just the filename:
```bash
python "Code/markovchain.py" --task harmonize --melody-input "my_melody.musicxml"
```

## Common Controls

- `--pitch-min`, `--pitch-max`: melody range
- `--max-jump`: max melodic leap
- `--key`, `--mode`: scale snapping
- `--duration-values`: duration quantization grid
- `--chord-repeat-penalty`: penalize repeated chords
- `--laplace-alpha`: smoothing strength

## Output

Generated MusicXML files:
- `generated_melody_*.musicxml`
- `generated_chords_*.musicxml`
- `generated_song_*.musicxml`

Open in MuseScore or any MusicXML-compatible editor.

## Notes

- First retrain can be slow due to symbolic parsing.
- Caching is enabled automatically; retrains become faster.
- Use `--refresh-cache` when source datasets change significantly.
