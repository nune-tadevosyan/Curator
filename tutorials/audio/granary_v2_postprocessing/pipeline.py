# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Granary v2 ASR postprocessing pipeline.

Reads ALM JSONL manifests, applies text cleaning and filtering, and writes
filtered output manifests mirroring the input directory structure.

Pipeline stages (per manifest):
  1. ALMManifestReader      — read JSONL manifest → one AudioTask per line
  2. InitializeFieldsStage  — copy pred_text → cleaned_text; skip_me = 0
  3. RegexSubstitutionStage — apply regex normalization rules to cleaned_text
  4. WhisperHallucinationStage — flag Whisper hallucination patterns
  5. FastTextLIDStage        — flag non-English or low-confidence transcriptions
  6. FinalizeFieldsStage     — text → v1_text; cleaned_text → text; drop pnc/itn/timestamp
  7. PreserveByValueStage    — drop entries where skip_me != 0
  8. ALMManifestWriterStage  — write surviving entries to mirrored output path

Usage::

    python tutorials/audio/granary_v2_postprocessing/pipeline.py \\
        --input_config /path/to/data_config.yaml \\
        --output_dir /path/to/output_root \\
        --fasttext_model lid.176.ftz
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.audio.alm.alm_manifest_writer import ALMManifestWriterStage
from nemo_curator.stages.audio.common import PreserveByValueStage
from nemo_curator.stages.audio.text_filtering import (
    FastTextLIDStage,
    FinalizeFieldsStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)

_TUTORIAL_DIR = Path(__file__).parent
_DEFAULT_REGEX_YAML = str(_TUTORIAL_DIR / "common.yaml")
_DEFAULT_HALL_PHRASES = str(_TUTORIAL_DIR / "en.txt")


def _compute_output_paths(manifest_paths: list[str], output_dir: str) -> dict[str, str]:
    """Mirror each input manifest path into output_dir, preserving relative structure.

    The common ancestor of all input manifests is stripped and the remainder
    is re-rooted under output_dir. For a single manifest the filename is
    preserved directly under output_dir.

    Example::

        input:  /data/results/batch_001/corpus_a/manifest_0.jsonl
        input:  /data/results/batch_002/corpus_b/manifest_1.jsonl
        output_dir: /out
        →  /out/batch_001/corpus_a/manifest_0.jsonl
           /out/batch_002/corpus_b/manifest_1.jsonl
    """
    if not manifest_paths:
        return {}
    paths = [Path(p) for p in manifest_paths]
    common = Path(os.path.commonpath([str(p) for p in paths]))
    # If common path is a file (single manifest), use its parent as anchor
    if common.is_file() or common.suffix:
        common = common.parent
    result: dict[str, str] = {}
    for p in paths:
        rel = p.relative_to(common)
        result[str(p)] = str(Path(output_dir) / rel)
    return result


def _create_pipeline(manifest_path: str, output_path: str, args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="Granary_v2_postprocessing",
        description=(
            "Text cleaning, hallucination detection, and language ID filtering "
            "for Granary v2 ASR manifests."
        ),
    )
    pipeline.add_stage(ALMManifestReader(manifest_path=manifest_path))
    pipeline.add_stage(InitializeFieldsStage())
    pipeline.add_stage(RegexSubstitutionStage(regex_params_yaml=args.regex_yaml))
    pipeline.add_stage(
        WhisperHallucinationStage(
            common_hall_file=args.hall_phrases,
            unique_words_threshold=args.unique_words_threshold,
            long_word_threshold=args.long_word_threshold,
            long_word_rel_threshold=args.long_word_rel_threshold,
        )
    )
    pipeline.add_stage(
        FastTextLIDStage(
            model_path=args.fasttext_model,
            target_lang=args.target_lang,
            min_lang_prob=args.min_lang_prob,
        )
    )
    pipeline.add_stage(FinalizeFieldsStage())
    pipeline.add_stage(PreserveByValueStage(input_value_key="skip_me", target_value=0, operator="eq"))
    pipeline.add_stage(ALMManifestWriterStage(output_path=output_path))
    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    with open(args.input_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    manifest_paths = [entry["manifest_filepath"] for entry in cfg["input_cfg"]]
    logger.info(f"Found {len(manifest_paths)} manifests in {args.input_config}")

    output_map = _compute_output_paths(manifest_paths, args.output_dir)
    for src, dst in output_map.items():
        logger.info(f"  {src}")
        logger.info(f"  → {dst}")

    executor = XennaExecutor()

    for i, (manifest_path, output_path) in enumerate(output_map.items(), 1):
        logger.info(f"\n[{i}/{len(output_map)}] Processing {manifest_path}")
        pipeline = _create_pipeline(manifest_path, output_path, args)
        if args.verbose:
            logger.debug(pipeline.describe())
        pipeline.run(executor)
        logger.info(f"  Written → {output_path}")

    logger.info(f"\nDone. {len(output_map)} manifest(s) written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Granary v2 ASR postprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_config",
        type=str,
        required=True,
        help="Path to YAML with input_cfg list (each entry must have a manifest_filepath key).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Input manifest paths are mirrored here.",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.ftz",
        help="FastText LID model: local path or known name (lid.176.bin / lid.176.ftz).",
    )
    parser.add_argument(
        "--regex_yaml",
        type=str,
        default=_DEFAULT_REGEX_YAML,
        help="Path to regex substitution rules YAML.",
    )
    parser.add_argument(
        "--hall_phrases",
        type=str,
        default=_DEFAULT_HALL_PHRASES,
        help="Path to hallucination phrases text file.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="en",
        help="Expected language code for LID filtering.",
    )
    parser.add_argument(
        "--min_lang_prob",
        type=float,
        default=0.3,
        help="Minimum FastText language probability to keep an entry.",
    )
    parser.add_argument(
        "--unique_words_threshold",
        type=float,
        default=0.4,
        help="Unique-word ratio threshold for repeated n-gram hallucination detection.",
    )
    parser.add_argument(
        "--long_word_threshold",
        type=int,
        default=25,
        help="Absolute character length above which a word is flagged as abnormally long.",
    )
    parser.add_argument(
        "--long_word_rel_threshold",
        type=float,
        default=3.0,
        help="Relative length ratio (longest/second-longest) for long-word hallucination detection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    main(parser.parse_args())
