# Markov Symbolic Music Generator

This repository generates symbolic music from MIDI/MusicXML using second-order Markov models.

Supported tasks:
- `melody`: melody only
- `chord`: chord progression only
- `song`: melody + chords together

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
