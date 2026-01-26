from __future__ import annotations

import json
from pathlib import Path

import torch

from python.inference.generate import (
    GoldenCase,
    _infer_device,
    generate,
    generate_golden_tests,
)


class _TinyTokenizer:
    def __init__(self, *, prompt_tokens: list[int], vocab_size: int = 16):
        self._prompt_tokens = list(prompt_tokens)
        self.vocab_size = int(vocab_size)

        self.newline_token_id = 1
        self.eos_token_id = 2

    def encode_inference_prompt(
        self, _prompt: str, *, width: int, height: int, style: str
    ) -> list[int]:
        _ = (width, height, style)
        return list(self._prompt_tokens)

    def decode(self, token_ids: list[int]) -> str:
        table = {
            0: "",
            1: "\n",
            2: "",
            3: "A",
            4: "B",
            5: "C",
            6: "D",
            7: "E",
            8: "F",
            9: "G",
        }
        return "".join(table.get(int(t), "?") for t in token_ids)

    def is_special_token(self, token_id: int) -> bool:
        return int(token_id) == 0


class _RecordingModel(torch.nn.Module):
    def __init__(self, *, vocab_size: int, next_token_id: int, block_size: int):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(()))
        self._vocab_size = int(vocab_size)
        self._next_token_id = int(next_token_id)
        self.seen_inputs: list[list[list[int]]] = []

        class _Config:
            def __init__(self, block_size: int):
                self.block_size = int(block_size)

        self.config = _Config(block_size)

    def forward(self, input_ids):
        self.seen_inputs.append(input_ids.detach().cpu().tolist())
        batch, seq = input_ids.shape
        logits = torch.full(
            (batch, seq, self._vocab_size),
            -10.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[:, :, self._next_token_id] = 10.0
        return logits, None


def test_infer_device_returns_cpu_when_parameters_unavailable() -> None:
    class _NoParams:
        def parameters(self):
            return []

    assert _infer_device(_NoParams()) == "cpu"


def test_infer_device_returns_cpu_when_parameters_raises() -> None:
    class _BadParams:
        def parameters(self):
            raise RuntimeError("boom")

    assert _infer_device(_BadParams()) == "cpu"


def test_generate_truncates_to_block_size() -> None:
    tok = _TinyTokenizer(prompt_tokens=list(range(10)), vocab_size=16)
    model = _RecordingModel(vocab_size=tok.vocab_size, next_token_id=3, block_size=4)

    out = generate(
        model,
        tok,
        "ignored",
        width=80,
        height=50,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=1,
        seed=0,
    )

    assert out == "A"
    assert model.seen_inputs, "Model was never called"
    assert model.seen_inputs[0][0] == [6, 7, 8, 9]


def test_generate_golden_tests_emits_valid_json(tmp_path: Path) -> None:
    tok = _TinyTokenizer(prompt_tokens=[0, 4, 5, 6], vocab_size=16)
    model = _RecordingModel(vocab_size=tok.vocab_size, next_token_id=4, block_size=32)

    out_dir = tmp_path / "goldens"
    cases = [GoldenCase(prompt="cat", width=4, height=2, style="art", seed=0)]
    paths = generate_golden_tests(model, tok, out_dir, cases=cases, device="cpu")

    assert paths == [out_dir / "golden_0.json"]
    payload = json.loads((out_dir / "golden_0.json").read_text(encoding="utf-8"))

    assert payload["case"]["prompt"] == "cat"
    assert payload["input_ids"] == [0, 4, 5, 6]
    assert payload["logits_shape"] == [1, 4, tok.vocab_size]
    assert payload["argmax_token"] == 4
    assert isinstance(payload["logits_sum"], float)
    assert len(payload["logits_first_10"]) == 10
