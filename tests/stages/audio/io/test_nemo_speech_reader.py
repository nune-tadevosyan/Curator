# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import gzip
import json
from typing import ClassVar

import pytest

from nemo_curator.stages.audio.io.nemo_speech_reader import (
    NeMoSpeechAudioReader,
    NeMoSpeechDiscoveryStage,
    NeMoSpeechReaderStage,
    _dedup_entries_by_stem,
    _load_input_cfg,
    _parse_input_cfg,
    _read_manifest_entries,
    _tar_member_name,
)
from nemo_curator.tasks import FileGroupTask


class TestDedupEntriesByStem:
    def test_keeps_preferred_format_for_same_stem(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/vid1.wav"},
            {"audio_filepath": "s3://b/audios/vid1.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        # .opus outranks .wav
        assert result[0]["audio_filepath"] == "s3://b/audios/vid1.opus"

    def test_preserves_order_and_distinct_stems(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/a.wav"},
            {"audio_filepath": "s3://b/audios/b.opus"},
            {"audio_filepath": "s3://b/audios/a.opus"},  # dup of a -> replaces, keeps a's position
            {"audio_filepath": "s3://b/audios/c.m4a"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        paths = [e["audio_filepath"] for e in result]
        assert paths == [
            "s3://b/audios/a.opus",
            "s3://b/audios/b.opus",
            "s3://b/audios/c.m4a",
        ]

    def test_no_duplicates_is_identity(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/x.opus"},
            {"audio_filepath": "s3://b/audios/y.wav"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 2

    def test_unknown_extension_ranks_last(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/audios/z.xyz"},
            {"audio_filepath": "s3://b/audios/z.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/audios/z.opus"

    def test_empty_paths_skipped(self) -> None:
        entries = [
            {"audio_filepath": ""},
            {"audio_filepath": "s3://b/audios/w.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/audios/w.opus"

    def test_keeps_distinct_recordings_sharing_basename_across_dirs(self) -> None:
        # Different directories, same basename -> genuinely distinct recordings.
        # Basename-only dedup would drop one; directory-aware dedup keeps both.
        entries = [
            {"audio_filepath": "s3://b/set_a/utt_001.wav"},
            {"audio_filepath": "s3://b/set_b/utt_001.wav"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        paths = [e["audio_filepath"] for e in result]
        assert paths == ["s3://b/set_a/utt_001.wav", "s3://b/set_b/utt_001.wav"]

    def test_same_dir_same_stem_still_dedups(self) -> None:
        entries = [
            {"audio_filepath": "s3://b/set_a/utt_001.wav"},
            {"audio_filepath": "s3://b/set_a/utt_001.opus"},
        ]
        result = _dedup_entries_by_stem(entries, "shard")
        assert len(result) == 1
        assert result[0]["audio_filepath"] == "s3://b/set_a/utt_001.opus"


class TestInlineInputCfg:
    """The reader accepts an inline ``input_cfg`` so no separate wrapper YAML is needed."""

    _INLINE: ClassVar[list] = [
        {
            "input_cfg": [
                {"corpus": "hi", "language": "hi", "type": "nemo", "manifest_filepath": "/data/hi/m.jsonl"},
            ]
        }
    ]

    def test_load_from_yaml_file(self, tmp_path) -> None:  # noqa: ANN001
        import yaml

        p = tmp_path / "data_config.yaml"
        p.write_text(yaml.safe_dump(self._INLINE), encoding="utf-8")
        assert _load_input_cfg(str(p), None) == self._INLINE

    def test_inline_takes_precedence_over_yaml(self) -> None:
        # yaml_path is a bogus path; inline is used, so no file read happens.
        assert _load_input_cfg("/does/not/exist.yaml", self._INLINE) == self._INLINE

    def test_neither_source_raises(self) -> None:
        with pytest.raises(ValueError, match="input_cfg or yaml_path"):
            _load_input_cfg(None, None)

    def test_parse_inline_produces_shard_descriptor(self) -> None:
        shards = _parse_input_cfg(self._INLINE, corpus_filter=None)
        assert shards == [{"corpus": "hi", "manifest_path": "/data/hi/m.jsonl", "language": "hi"}]

    def test_parse_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match="input_cfg list"):
            _parse_input_cfg({"not": "a list"}, corpus_filter=None)

    def test_omegaconf_interpolation_is_resolved(self) -> None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "input_manifest": "/data/hi/m.jsonl",
                "data_config": [
                    {"input_cfg": [{"corpus": "hi", "language": "hi", "manifest_filepath": "${input_manifest}"}]}
                ],
            }
        )
        resolved = _load_input_cfg(None, cfg.data_config)
        shards = _parse_input_cfg(resolved, corpus_filter=None)
        assert shards[0]["manifest_path"] == "/data/hi/m.jsonl"

    def test_flat_cfg_without_input_cfg_key(self) -> None:
        # A plain list of cfg dicts (no wrapping ``input_cfg`` key) is also accepted.
        flat = [{"corpus": "hi", "language": "hi", "manifest_filepath": "/data/hi/m.jsonl"}]
        shards = _parse_input_cfg(flat, corpus_filter=None)
        assert shards == [{"corpus": "hi", "manifest_path": "/data/hi/m.jsonl", "language": "hi"}]

    def test_reader_requires_a_source(self) -> None:
        with pytest.raises(ValueError, match="input_cfg or yaml_path"):
            NeMoSpeechAudioReader()

    def test_reader_accepts_inline_cfg(self) -> None:
        reader = NeMoSpeechAudioReader(input_cfg=self._INLINE)
        discovery = reader.decompose()[0]
        assert isinstance(discovery, NeMoSpeechDiscoveryStage)
        assert discovery.input_cfg == self._INLINE

    def test_emit_chunk_size_reaches_reader_stage(self) -> None:
        reader = NeMoSpeechAudioReader(input_cfg=self._INLINE, emit_chunk_size=4)
        assert reader.decompose()[-1].emit_chunk_size == 4


class TestTarMemberName:
    def test_plain_path_is_unchanged(self) -> None:
        assert _tar_member_name("utt_001.wav") == "utt_001.wav"

    def test_offset_sub_segments_share_one_member(self) -> None:
        assert _tar_member_name("utt_001-sub1.wav") == "utt_001.wav"
        assert _tar_member_name("utt_001-sub12.wav") == "utt_001.wav"

    def test_suffix_without_extension(self) -> None:
        assert _tar_member_name("utt_001-sub3") == "utt_001"


class TestReadManifestEntries:
    def test_reads_plain_jsonl(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "m.jsonl"
        path.write_text('{"audio_filepath": "a.wav"}\n\n{"audio_filepath": "b.wav"}\n', encoding="utf-8")
        assert _read_manifest_entries(str(path)) == [{"audio_filepath": "a.wav"}, {"audio_filepath": "b.wav"}]

    def test_reads_gzipped_jsonl(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "m.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write('{"audio_filepath": "a.wav"}\n')
        assert _read_manifest_entries(str(path)) == [{"audio_filepath": "a.wav"}]


class _FakeCut:
    """Minimal stand-in for a lhotse Cut, enough for the reader's shard loop."""

    def __init__(self, cut_id: str, recording_id: str | None = None) -> None:
        self.id = cut_id
        self.recording_id = recording_id or cut_id


class TestStreamCutset:
    """The tarred reader emits a shard in chunks instead of buffering all of it.

    The tar is a forward-only stream, so these tests stub the CutSet and the per-cut
    decode: what matters is the chunking, the shard total, and the audit rows that keep
    the writer's ``.done`` accounting exact.
    """

    @staticmethod
    def _write_manifest(tmp_path, entries: list[dict]) -> str:  # noqa: ANN001
        path = tmp_path / "manifest_0.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        return str(path)

    @staticmethod
    def _stage(monkeypatch: pytest.MonkeyPatch, cuts: list[_FakeCut], **kwargs) -> NeMoSpeechReaderStage:
        stage = NeMoSpeechReaderStage(**kwargs)
        monkeypatch.setattr(stage, "_make_cutset", lambda *_args, **_kw: list(cuts))
        monkeypatch.setattr(
            stage,
            "_build_cut_entry",
            lambda cut, corpus, _language: {"audio_filepath": cut.id, "corpus": corpus},
        )
        return stage

    @staticmethod
    def _task(manifest_path: str) -> FileGroupTask:
        return FileGroupTask(
            task_id="shard_0",
            dataset_name="corp",
            data=[manifest_path, "/data/audio_0.tar"],
            reader_config={"corpus": "corp", "shard_key": "corp/shard_0", "language": "en"},
        )

    def test_yields_chunks_of_emit_chunk_size(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        names = [f"utt_{i}.wav" for i in range(5)]
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = self._stage(monkeypatch, [_FakeCut(n) for n in names], emit_chunk_size=2)

        chunks = list(stage._stream_cutset(self._task(manifest)))

        assert [len(c) for c in chunks] == [2, 2, 1]

    def test_shard_total_is_known_on_the_first_chunk(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        # The writer only writes .done once it has seen _shard_total rows, so the very
        # first emitted task must already carry the full count.
        names = [f"utt_{i}.wav" for i in range(5)]
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = self._stage(monkeypatch, [_FakeCut(n) for n in names], emit_chunk_size=2)

        first_chunk = next(iter(stage._stream_cutset(self._task(manifest))))

        assert all(t._metadata["_shard_total"] == 5 for t in first_chunk)
        assert all(t._metadata["_shard_key"] == "corp/shard_0" for t in first_chunk)

    def test_emitted_count_matches_shard_total(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        names = [f"utt_{i}.wav" for i in range(5)]
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = self._stage(monkeypatch, [_FakeCut(n) for n in names], emit_chunk_size=2)

        tasks = stage._process_cutset(self._task(manifest))

        assert len(tasks) == 5
        assert [t.data["audio_filepath"] for t in tasks] == names

    def test_missing_tar_members_become_read_error_rows(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        # NeMo's adapter silently skips members absent from (or corrupt in) the tar.
        # Placeholders keep the row count at _shard_total so .done still fires.
        names = [f"utt_{i}.wav" for i in range(5)]
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = self._stage(monkeypatch, [_FakeCut(n) for n in names[:3]], emit_chunk_size=2)

        tasks = stage._process_cutset(self._task(manifest))

        assert len(tasks) == 5
        read_errors = [t for t in tasks if t.data.get("read_error")]
        assert len(read_errors) == 2
        assert {t.data["audio_filepath"] for t in read_errors} == {"utt_3.wav", "utt_4.wav"}

    def test_skipme_entries_are_excluded_from_shard_total(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        # The adapter never yields cuts for _skipme entries, so counting them would
        # leave the shard permanently one row short of completion.
        manifest = self._write_manifest(
            tmp_path,
            [
                {"audio_filepath": "utt_0.wav"},
                {"audio_filepath": "utt_1.wav", "_skipme": "bad audio"},
                {"audio_filepath": "utt_2.wav"},
            ],
        )
        stage = self._stage(monkeypatch, [_FakeCut("utt_0.wav"), _FakeCut("utt_2.wav")])

        tasks = stage._process_cutset(self._task(manifest))

        assert len(tasks) == 2
        assert all(t._metadata["_shard_total"] == 2 for t in tasks)
        assert not any(t.data.get("read_error") for t in tasks)

    def test_offset_sub_segments_do_not_produce_placeholders(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        # Two manifest rows, one tar member: both cuts come from recording "utt_0.wav".
        manifest = self._write_manifest(
            tmp_path,
            [
                {"audio_filepath": "utt_0-sub1.wav", "offset": 0.0},
                {"audio_filepath": "utt_0-sub2.wav", "offset": 1.0},
            ],
        )
        cuts = [_FakeCut("utt_0-sub1.wav", recording_id="utt_0.wav"), _FakeCut("utt_0-sub2.wav", "utt_0.wav")]
        stage = self._stage(monkeypatch, cuts)

        tasks = stage._process_cutset(self._task(manifest))

        assert len(tasks) == 2
        assert not any(t.data.get("read_error") for t in tasks)
        assert all(t._metadata["_shard_total"] == 2 for t in tasks)

    def test_undecodable_cut_is_replaced_by_a_read_error_row(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        names = ["utt_0.wav", "utt_1.wav"]
        manifest = self._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = NeMoSpeechReaderStage()
        monkeypatch.setattr(stage, "_make_cutset", lambda *_a, **_kw: [_FakeCut(n) for n in names])

        def explode_on_second(cut, corpus, _language):  # noqa: ANN001, ANN202
            if cut.id == "utt_1.wav":
                msg = "corrupt audio"
                raise RuntimeError(msg)
            return {"audio_filepath": cut.id, "corpus": corpus}

        monkeypatch.setattr(stage, "_build_cut_entry", explode_on_second)

        tasks = stage._process_cutset(self._task(manifest))

        assert len(tasks) == 2
        assert tasks[1].data["read_error"] is True
        assert tasks[1].data["audio_filepath"] == "utt_1.wav"

    def test_unreadable_manifest_falls_back_to_buffering_whole_shard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Without a readable manifest the total is only known after the last cut, so the
        # shard is emitted as a single chunk with the count observed during the read.
        names = [f"utt_{i}.wav" for i in range(3)]
        stage = self._stage(monkeypatch, [_FakeCut(n) for n in names], emit_chunk_size=1)

        chunks = list(stage._stream_cutset(self._task("/nonexistent/manifest_0.jsonl")))

        assert [len(c) for c in chunks] == [3]
        assert all(t._metadata["_shard_total"] == 3 for t in chunks[0])


class TestReaderStreamingContract:
    """``process_batch`` must stay equivalent to draining ``process_stream``.

    Backends without incremental output (Xenna, Ray actor pool) call ``process_batch``,
    so the two paths have to produce the same tasks in the same order.
    """

    def test_reader_declares_streaming_support(self) -> None:
        assert NeMoSpeechReaderStage().supports_streaming() is True

    def test_non_streaming_stage_does_not_declare_support(self) -> None:
        assert NeMoSpeechDiscoveryStage(yaml_path="/some/config.yaml").supports_streaming() is False

    def test_process_batch_matches_process_stream(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
        names = [f"utt_{i}.wav" for i in range(5)]
        manifest = TestStreamCutset._write_manifest(tmp_path, [{"audio_filepath": n} for n in names])
        stage = TestStreamCutset._stage(monkeypatch, [_FakeCut(n) for n in names], emit_chunk_size=2)
        task = TestStreamCutset._task(manifest)

        batched = stage.process_batch([task])
        streamed = [t for chunk in stage.process_stream([task]) for t in chunk]

        assert [t.task_id for t in batched] == [t.task_id for t in streamed]
        assert len(batched) == 5
