# Granary v2 ASR Postprocessing Pipeline

Postprocessing pipeline for Granary v2 ASR manifests. Reads JSONL manifests produced by ASR inference, cleans and filters transcriptions based on text quality, and writes a filtered output manifest.

## What it does

Each manifest entry is processed through the following stages:

| # | Stage | Description |
|---|---|---|
| 1 | `ALMManifestReader` | Reads JSONL manifests — one `AudioTask` per line |
| 2 | `InitializeFieldsStage` | Copies `pred_text` → `cleaned_text`; sets `skip_me = 0` |
| 3 | `RegexSubstitutionStage` | Applies regex normalization rules (quotes, dashes, brackets, character filtering, whitespace) |
| 4 | `WhisperHallucinationStage` | Flags entries with repeated n-grams, abnormally long words, or known hallucination phrases |
| 5 | `FastTextLIDStage` | Flags non-English transcriptions or low-confidence language ID predictions |
| 6 | `FinalizeFieldsStage` | Renames `text` → `v1_text`, promotes `cleaned_text` → `text`, drops `pnc`/`itn`/`timestamp` |
| 7 | `PreserveByValueStage` | Drops all entries where `skip_me = 1` |
| 8 | `ALMManifestWriterStage` | Writes surviving entries to the output JSONL |

Entries that fail any filter step are marked `skip_me = 1` and dropped at the end. The original `pred_text` field is always preserved in the output alongside the cleaned `text`.

## Input format

The `--input_config` YAML must have a top-level `input_cfg` list. Each entry needs a `manifest_filepath` key pointing to a JSONL manifest:

```yaml
input_cfg:
  - corpus: librilight
    manifest_filepath: /path/to/manifest_0.jsonl
    hours: 120.0
  - corpus: ami
    manifest_filepath: /path/to/ami_manifest.jsonl
    hours: 80.0
```

Each manifest line is a JSON dict with at minimum an `audio_filepath` and `pred_text` field.

## Bundled config files

`common.yaml` (regex rules) and `en.txt` (hallucination phrases) are bundled in this directory and used by default — no need to pass them as arguments.

## Prerequisites

Install the audio extras:

```bash
uv sync --extra audio_cuda12
```

The FastText LID model (`lid.176.ftz`) is downloaded automatically on first run to `~/.cache/nemo_curator/fasttext/`. To use a local copy, pass its path via `--fasttext_model`.

## Usage

```bash
python tutorials/audio/granary_v2_postprocessing/pipeline.py \
    --input_config /path/to/data_config.yaml \
    --output_dir /path/to/output_root \
    --fasttext_model lid.176.ftz
```

The pipeline runs once per manifest and mirrors the input directory structure under `--output_dir`. For example:

```
input:   /data/results/batch_001/corpus_a/manifest_0.jsonl
input:   /data/results/batch_002/corpus_b/manifest_1.jsonl

output:  /path/to/output_root/batch_001/corpus_a/manifest_0.jsonl
         /path/to/output_root/batch_002/corpus_b/manifest_1.jsonl
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input_config` | yes | — | Path to YAML with `input_cfg` list |
| `--output_dir` | yes | — | Root output directory; input structure is mirrored here |
| `--fasttext_model` | no | `lid.176.ftz` (auto-downloaded) | Local model path or `lid.176.bin` / `lid.176.ftz` |
| `--regex_yaml` | no | `common.yaml` (bundled) | Regex substitution rules YAML |
| `--hall_phrases` | no | `en.txt` (bundled) | Hallucination phrases file |
| `--target_lang` | no | `en` | Expected language code for LID filtering |
| `--min_lang_prob` | no | `0.3` | Minimum FastText confidence to keep an entry |
| `--unique_words_threshold` | no | `0.4` | Max unique-word ratio before flagging repeated n-grams |
| `--long_word_threshold` | no | `25` | Character length above which a word is considered abnormally long |
| `--long_word_rel_threshold` | no | `3.0` | Longest/second-longest word ratio for long-word detection |
| `--verbose` | no | off | Enable DEBUG logging |

## Output schema

Each surviving entry contains:

| Field | Source |
|---|---|
| `text` | Cleaned and normalized transcription (was `cleaned_text`) |
| `v1_text` | Original reference text from the input manifest |
| `pred_text` | Raw ASR prediction (unchanged) |
| `audio_filepath` | Path to audio file |
| `duration` | Audio duration in seconds |
| All other original fields | Preserved as-is (except `pnc`, `itn`, `timestamp` which are dropped) |

## Stage implementation

The five new stages live in `nemo_curator/stages/audio/text_filtering/` and are exported from `nemo_curator.stages.audio`:

```python
from nemo_curator.stages.audio import (
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
    FastTextLIDStage,
    FinalizeFieldsStage,
)
```

They can be used independently in any custom pipeline that processes `AudioTask` data.
