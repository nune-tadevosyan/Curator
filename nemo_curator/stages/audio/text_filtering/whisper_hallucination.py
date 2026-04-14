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

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


@dataclass
class WhisperHallucinationStage(ProcessingStage[AudioTask, AudioTask]):
    """Detect common Whisper hallucination patterns and flag entries with skip_me=1.

    Three checks are applied:
    - Repeated n-grams: low lexical diversity (unique-word ratio <= threshold).
    - Long word: an abnormally long word or a word much longer than its neighbours.
    - Frequent single phrase: the full transcript matches a known hallucination phrase.

    If any check triggers, ``skip_me`` is set to 1 (existing value of 1 is preserved).
    No intermediate flag fields are added to the task.
    """

    common_hall_file: str = ""
    unique_words_threshold: float = 0.4
    long_word_threshold: int = 25
    long_word_rel_threshold: float = 3.0
    text_key: str = "cleaned_text"
    skip_me_key: str = "skip_me"
    name: str = "WhisperHallucination"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _phrases: set[str] = field(default_factory=set, init=False, repr=False)
    _setup_called: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.common_hall_file:
            msg = "common_hall_file is required for WhisperHallucinationStage"
            raise ValueError(msg)

    def setup(self, worker_metadata: Any = None) -> None:
        phrases: set[str] = set()
        with open(self.common_hall_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Strip trailing frequency count (integer, possibly negative)
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        int(parts[1])
                        phrases.add(parts[0])
                    except ValueError:
                        phrases.add(line)
                else:
                    phrases.add(line)
        self._phrases = phrases
        self._setup_called = True
        logger.info(f"WhisperHallucinationStage: loaded {len(phrases)} phrases from {self.common_hall_file}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key]

    def _repeated_ngrams(self, words: list[str]) -> bool:
        if not words:
            return False
        return len(set(words)) / len(words) <= self.unique_words_threshold

    def _long_word(self, words: list[str]) -> bool:
        if not words:
            return False
        lengths = sorted(len(w) for w in words)
        if lengths[-1] >= self.long_word_threshold:
            return True
        if len(lengths) > 1 and lengths[-2] > 0:
            return (lengths[-1] - lengths[-2]) / lengths[-2] >= self.long_word_rel_threshold
        return False

    def _frequent_single_word(self, text: str) -> bool:
        cleaned = text.strip().rstrip(".,?!")
        return cleaned in self._phrases

    def process(self, task: AudioTask) -> AudioTask:
        if not self._setup_called:
            logger.warning(
                f"WhisperHallucinationStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        words = text.split()
        flagged = self._repeated_ngrams(words) or self._long_word(words) or self._frequent_single_word(text)
        if flagged:
            task.data[self.skip_me_key] = 1
        return task
