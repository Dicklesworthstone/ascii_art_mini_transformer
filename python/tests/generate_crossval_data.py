#!/usr/bin/env python3
"""
Generate cross-validation golden data for Python vs Rust comparison.

This script creates the test data files used by Rust cross-validation tests.
Run from the project root:
    python3 python/tests/generate_crossval_data.py

Output files are written to: rust/ascii-gen/test_data/crossval/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from python.model.tokenizer import AsciiTokenizer  # noqa: E402
from python.model.transformer import AsciiGPT, AsciiGPTConfig  # noqa: E402
from python.inference.generate import generate_golden_tests, GoldenCase  # noqa: E402


def main() -> int:
    output_dir = project_root / "rust/ascii-gen/test_data/crossval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating cross-validation data...")

    # Create a small test model with fixed seed for reproducibility
    torch.manual_seed(42)

    config = AsciiGPTConfig(
        vocab_size=107,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        max_rows=32,
        max_cols=64,
    )

    model = AsciiGPT(config)
    model.eval()

    tokenizer = AsciiTokenizer()

    # 1. Generate model golden tests (logits comparison)
    print("\n1. Generating model golden tests...")
    golden_dir = output_dir / "golden"
    golden_dir.mkdir(exist_ok=True)

    cases = [
        GoldenCase(prompt="cat", width=40, height=20, style="art", seed=0),
        GoldenCase(prompt="star", width=20, height=10, style="simple", seed=1),
        GoldenCase(prompt="HELLO", width=80, height=8, style="banner", seed=2),
        GoldenCase(
            prompt="hello, world!", width=128, height=64, style="detailed", seed=3
        ),
        GoldenCase(prompt="wrap", width=4, height=10, style="art", seed=4),
    ]

    paths = generate_golden_tests(
        model, tokenizer, golden_dir, cases=cases, device="cpu"
    )
    for p in paths:
        print(f"  Created: {p}")

    # 2. Generate tokenizer golden tests
    print("\n2. Generating tokenizer golden tests...")

    tokenizer_golden = {
        "encode_inference_prompt": [],
        "encode_text": [],
    }

    # Prompt encoding tests
    prompt_cases = [
        ("cat", 40, 20, "art"),
        ("star", 20, 10, "simple"),
        ("HELLO", 80, 8, "banner"),
        ("hello, world!", 128, 64, "detailed"),
        ("wrap", 4, 10, "art"),
    ]
    for prompt, width, height, style in prompt_cases:
        ids = tokenizer.encode_inference_prompt(
            prompt, width=width, height=height, style=style
        )
        tokenizer_golden["encode_inference_prompt"].append(
            {
                "prompt": prompt,
                "width": width,
                "height": height,
                "style": style,
                "ids": ids,
            }
        )

    # Text encoding tests
    text_cases = [
        "Hello World",
        "hello, world!",
        "!@#$%^&*()",
        "ASCII",
        "",
    ]
    for text in text_cases:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        tokenizer_golden["encode_text"].append(
            {
                "text": text,
                "ids": ids,
                "decoded": decoded,
            }
        )

    tokenizer_path = output_dir / "tokenizer_golden.json"
    tokenizer_path.write_text(json.dumps(tokenizer_golden, indent=2))
    print(f"  Created: {tokenizer_path}")

    # 3. Generate greedy generation golden tests
    print("\n3. Generating greedy generation golden tests...")

    from python.inference.constraints import ConstrainedDecoder

    greedy_golden = []

    # Define output token filter to match Rust behavior
    # Rust token structure:
    #   0: PAD, 1: BOS, 2: EOS, 3: UNK, 4: SEP, 5: WIDTH, 6: HEIGHT, 7: NEWLINE
    #   8-11: Style tokens (STYLE_ART, STYLE_BANNER, STYLE_SIMPLE, STYLE_DETAILED)
    #   12-106: Printable ASCII (space through tilde)
    NEWLINE_ID = 7
    PRINTABLE_ASCII_START = 12
    PRINTABLE_ASCII_END = 106

    def is_output_token(token_id: int) -> bool:
        """Check if a token is an output token (newline or printable ASCII)."""
        return token_id == NEWLINE_ID or (
            PRINTABLE_ASCII_START <= token_id <= PRINTABLE_ASCII_END
        )

    def mask_non_output_tokens(logits_tensor, tok):
        """Mask non-output tokens (control tokens) to -inf to match Rust behavior."""
        result = logits_tensor.clone()
        for i in range(tok.vocab_size):
            # Skip EOS - it's always allowed
            if i == tok.eos_token_id:
                continue
            # Mask non-output tokens
            if not is_output_token(i):
                result[i] = float("-inf")
        return result

    for case in cases:
        torch.manual_seed(case.seed)

        input_ids = tokenizer.encode_inference_prompt(
            case.prompt, width=case.width, height=case.height, style=case.style
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        decoder = ConstrainedDecoder(
            max_width=case.width,
            max_height=case.height,
            max_tokens=32,  # Limited for test purposes
        )

        generated = []
        max_new_tokens = 16  # Generate a small fixed number for testing

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = model(input_tensor)
                next_logits = logits[0, -1, :]

            # First mask non-output tokens (to match Rust behavior)
            next_logits = mask_non_output_tokens(next_logits, tokenizer)
            # Then apply constraint-based masking
            next_logits = decoder.apply_constraints_to_logits(next_logits, tokenizer)

            # Greedy: temperature=0
            next_token = int(torch.argmax(next_logits).item())

            if next_token == tokenizer.eos_token_id:
                generated.append(next_token)
                break

            generated.append(next_token)
            decoder.update(next_token, tokenizer)

            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], dtype=torch.long)], dim=1
            )

            if decoder.should_stop(tokenizer):
                break

        greedy_golden.append(
            {
                "case": {
                    "prompt": case.prompt,
                    "width": case.width,
                    "height": case.height,
                    "style": case.style,
                    "seed": case.seed,
                },
                "prompt_ids": input_ids,
                "generated_ids": generated,
                "full_sequence": input_ids + generated,
            }
        )

    greedy_path = output_dir / "greedy_golden.json"
    greedy_path.write_text(json.dumps(greedy_golden, indent=2))
    print(f"  Created: {greedy_path}")

    # 4. Export model weights
    print("\n4. Exporting model weights...")
    from safetensors.torch import save_file

    weights_path = output_dir / "model.safetensors"
    state_dict = model.state_dict()
    export_dict = {}
    for name, tensor in state_dict.items():
        # Skip attention mask buffers (deterministic, rebuilt at runtime)
        if name.endswith(".mask"):
            continue
        export_dict[name] = tensor.detach().float().contiguous()

    # Break weight-tying (token_embedding.weight == lm_head.weight) by cloning
    if "token_embedding.weight" in export_dict and "lm_head.weight" in export_dict:
        export_dict["lm_head.weight"] = export_dict["lm_head.weight"].clone()

    save_file(export_dict, weights_path)
    print(f"  Created: {weights_path}")

    # 5. Export config
    print("\n5. Exporting config...")
    config_dict = {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "max_rows": config.max_rows,
        "max_cols": config.max_cols,
        "newline_token_id": tokenizer.newline_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "export_dtype": "float32",
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config_dict, indent=2))
    print(f"  Created: {config_path}")

    # 6. Export tokenizer config
    print("\n6. Exporting tokenizer config...")
    tokenizer_config_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_config_path)
    print(f"  Created: {tokenizer_config_path}")

    print("\n" + "=" * 60)
    print("Cross-validation data generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
