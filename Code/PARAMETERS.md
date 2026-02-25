# Markov Music Generator Parameters

This file documents all runtime parameters for:

- `12. Melody generation with Markov chains/Code/markovchain.py`

## Quick Usage

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" [options]
```

## Main Modes

- `--task melody`
  - Generate melody only.
- `--task chord`
  - Generate chord progression only.
- `--task song`
  - Generate melody + chords together in one score.
- `--task realtime`
  - Interactive bar-by-bar chorale generation.
- `--task harmonize`
  - Harmonize an input melody into SATB.

## Train / Load Control

- `--retrain`
  - Force retraining and overwrite saved model(s).
  - If omitted, existing model file(s) are loaded if present.

- `--refresh-cache`
  - Rebuild parsing cache from source files instead of reusing cached events.

## Shared Parameters (all tasks)

- `--laplace-alpha <float>` (default: `1.0`)
  - Smoothing strength for Markov transitions.

- `--seed-midi <int>` (default: `None`)
  - Optional melody start pitch (MIDI 0-127).

## Dataset / Cache / Model Paths

- `--data-folder <path>`
  - Training folder for `melody` or `chord` task.
  - Default by task:
    - `melody` -> `data`
    - `chord` -> `dataChords`

- `--cache-path <path>`
  - Event cache location.
  - Default by task:
    - `melody` -> `cache/note_events_cache.json`
    - `chord` -> `cache/chord_events_cache.json`

- `--model-path <path>`
  - Model file for single-task runs.
  - Default by task:
    - `melody` -> `models/factorized_markov_model.npz`
    - `chord` -> `models/chord_markov_model.npz`

## Melody Controls (`--task melody`)

- `--length <int>` (default: `40`)
  - Number of generated melody notes.

- `--pitch-min <int>` (default: `48`)
  - Lowest allowed MIDI pitch.

- `--pitch-max <int>` (default: `84`)
  - Highest allowed MIDI pitch.

- `--max-jump <int>` (default: `7`)
  - Maximum semitone jump between consecutive melody notes.

- `--key <str>` (default: `C`)
  - Scale root for snapping, e.g. `C`, `G`, `F#`, `Bb`.

- `--mode {major,minor}` (default: `major`)
  - Scale type for snapping.

- `--disable-scale-snap`
  - Turn off scale snapping.

- `--duration-values "<csv>"` (default: `"0.25,0.5,1.0,2.0"`)
  - Allowed duration grid for quantization.

- `--disable-duration-quantize`
  - Turn off duration quantization.

## Chord Controls (`--task chord`)

- `--length <int>` (default: `40`)
  - Number of generated chord events.

- `--duration-values "<csv>"` (default: `"0.25,0.5,1.0,2.0"`)
  - Duration quantization grid for chord events.

- `--disable-duration-quantize`
  - Turn off duration quantization.

- `--chord-octave <int>` (default: `4`)
  - Octave used for rendering generated block chords.

- `--chord-repeat-penalty <float>` (default: `0.5`)
  - Penalizes consecutive repetition of the same chord identity.
  - Range is effectively clamped to `0.0` to `0.99`.
  - `0.0` means no penalty.

## Song Controls (`--task song`)

- `--song-bars <int>` (default: `8`)
  - Number of bars to generate.

- `--beats-per-bar <int>` (default: `4`)
  - Melody note slots per bar.

- `--song-style {lead,chorale}` (default: `lead`)
  - `lead`: original melody + block chords layout.
  - `chorale`: SATB 4-voice harmony conditioned on generated chord progression.

- `--chorale-beam-width <int>` (default: `20`)
  - Beam width for SATB search in `chorale` mode.
  - Higher values are slower but usually produce smoother results.

- `--chorale-candidates-per-voice <int>` (default: `6`)
  - Number of ranked pitch candidates per voice before combining SATB sonorities.

- `--chorale-top-sonorities <int>` (default: `24`)
  - Number of top local SATB sonorities kept per beam item at each time step.

- `--cadence-every-bars <int>` (default: `4`)
  - Forces a V-I cadence every N bars in `song` modes.
  - Use `1` or `0` to effectively disable cadence forcing.

- `--disable-progression-blocks`
  - Disables progression-block transitions in `song` modes.
  - Uses direct chord-chain generation only.

- `--progression-laplace-alpha <float>` (default: `1.0`)
  - Laplace smoothing used by the progression-block transition model.
  - Progression blocks are extracted from `dataChords` and split into major/minor sets.

- `--chorale-repeat-note-penalty <float>` (default: `5.0`)
  - Base penalty for repeating the same note in the same voice in chorale mode.
  - Additional streak penalty is applied for consecutive repeats.

- `--melody-model-path <path>` (default: `models/factorized_markov_model.npz`)
  - Melody model path used in song mode.

- `--chord-model-path <path>` (default: `models/chord_markov_model.npz`)
  - Chord model path used in song mode.

- `--melody-data-folder <path>` (default: `data`)
  - Melody training dataset when retraining in song mode.

- `--chord-data-folder <path>` (default: `dataChords`)
  - Chord training dataset when retraining in song mode.

- `--realtime-bars <int>` (default: `16`)
  - Number of bars generated in realtime mode before auto-stop.

- `--disable-realtime-live-update`
  - Disable per-bar live MusicXML snapshot updates in realtime mode.

- `--realtime-open-each-bar`
  - Re-open the live MusicXML file every bar update (can open many tabs/windows).

- `--realtime-live-file <path>` (default: `generated_realtime_live.musicxml`)
  - MusicXML file path updated after each realtime bar.

- `--melody-input <path>`
  - Required for `--task harmonize`.
  - Input melody file (MIDI/MusicXML) used as soprano guide.

- `--melody-input-folder <path>` (default: `melodyInput`)
  - Default folder searched for `--melody-input` when direct path is not found.

- Melody-shaping options also apply in song mode:
  - `--seed-midi`
  - `--pitch-min`
  - `--pitch-max`
  - `--max-jump`
  - `--key`
  - `--mode`
  - `--disable-scale-snap`
  - `--chord-octave`
  - `--chord-repeat-penalty`

## Useful Command Examples

### 1) Retrain melody and generate

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" --task melody --retrain --refresh-cache
```

### 2) Retrain chords from `dataChords` and generate progression

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" --task chord --retrain --refresh-cache --length 16
```

### 3) Generate full song from saved models

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" --task song --song-bars 8 --beats-per-bar 4
```

### 4) Generate SATB chorale song

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" --task song --song-style chorale --song-bars 8 --beats-per-bar 4
```

### 5) Song with tighter melody constraints

```powershell
python "12. Melody generation with Markov chains/Code/markovchain.py" --task song --pitch-min 55 --pitch-max 79 --max-jump 5 --key C --mode major
```
