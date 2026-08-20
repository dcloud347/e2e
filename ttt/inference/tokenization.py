"""Tokenizer and batching helpers for raw text evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ttt.model.data import Batch

FinalWindowPolicy = Literal["pad", "drop", "error"]


@dataclass(frozen=True)
class TokenWindow:
    """A fixed-length token window and its LM batch."""

    index: int
    tokens: np.ndarray
    batch: Batch
    valid_targets: int


def load_tokenizer(tokenizer_name_or_path: str, *, token: str | None = None):
    """Load a Hugging Face tokenizer lazily so tests do not require transformers."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Raw text eval requires `transformers`. Run `uv sync --exact` after updating dependencies.") from exc

    kwargs = {"use_fast": True}
    if token:
        kwargs["token"] = token
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, **kwargs)


def validate_tokenizer(
    tokenizer,
    *,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    strict: bool = True,
) -> list[str]:
    """Validate tokenizer/model compatibility and return warnings."""

    warnings: list[str] = []

    tokenizer_size = len(tokenizer)
    if tokenizer_size != vocab_size:
        message = f"Tokenizer size {tokenizer_size} does not match model vocab_size {vocab_size}."
        if strict:
            raise ValueError(message)
        warnings.append(message)

    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id != bos_token_id:
        message = f"Tokenizer bos_token_id {tokenizer.bos_token_id} does not match model bos_token_id {bos_token_id}."
        if strict:
            raise ValueError(message)
        warnings.append(message)

    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id != eos_token_id:
        message = f"Tokenizer eos_token_id {tokenizer.eos_token_id} does not match model eos_token_id {eos_token_id}."
        if strict:
            raise ValueError(message)
        warnings.append(message)

    return warnings


def encode_text(
    tokenizer,
    text: str,
    *,
    bos_token_id: int,
    eos_token_id: int,
    add_bos: bool = True,
    add_eos: bool = False,
) -> np.ndarray:
    """Encode text and handle BOS/EOS explicitly."""

    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if add_bos:
        token_ids.insert(0, bos_token_id)
    if add_eos:
        token_ids.append(eos_token_id)
    return np.asarray(token_ids, dtype=np.int32)


def read_text_input(*, text: str | None, text_file: str | None) -> str:
    """Read text from exactly one CLI input source."""

    if (text is None) == (text_file is None):
        raise ValueError("Pass exactly one of `--text` or `--text-file`.")
    if text is not None:
        return text
    return Path(text_file).read_text()


def tokens_to_lm_batch(
    tokens: np.ndarray,
    *,
    seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int | None = None,
    pad: bool = False,
) -> Batch:
    """Convert up to `seq_len + 1` token ids into the repository LM Batch format."""

    tokens = np.asarray(tokens, dtype=np.int32)
    if tokens.ndim != 1:
        raise ValueError(f"Expected a 1D token array, got shape {tokens.shape}.")
    if len(tokens) < 2:
        raise ValueError("Need at least two tokens to build next-token targets.")
    if len(tokens) > seq_len + 1:
        raise ValueError(f"Got {len(tokens)} tokens for a {seq_len + 1}-token window.")

    original_len = len(tokens)
    if original_len < seq_len + 1:
        if not pad:
            raise ValueError(f"Need {seq_len + 1} tokens for seq_len={seq_len}; got {original_len}.")
        pad_id = eos_token_id if pad_token_id is None else pad_token_id
        tokens = np.pad(tokens, (0, seq_len + 1 - original_len), constant_values=pad_id).astype(np.int32)

    input_ids = tokens[:-1]
    target_tokens = tokens[1:]
    loss_masks = target_tokens != bos_token_id

    if original_len < seq_len + 1:
        valid = np.zeros(seq_len, dtype=bool)
        valid[: original_len - 1] = True
        loss_masks = loss_masks & valid

    return Batch(
        input_ids=input_ids.astype(np.int32),
        target_tokens=target_tokens.astype(np.int32),
        loss_masks=loss_masks,
    )


def iter_token_windows(
    tokens: np.ndarray,
    *,
    seq_len: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int | None = None,
    final_window: FinalWindowPolicy = "pad",
    max_windows: int | None = None,
) -> Iterator[TokenWindow]:
    """Yield fixed-length LM windows from a raw token sequence."""

    tokens = np.asarray(tokens, dtype=np.int32)
    if len(tokens) < 2:
        raise ValueError("Need at least two tokens to evaluate text.")

    window_size = seq_len + 1
    window_index = 0

    for start in range(0, len(tokens) - 1, seq_len):
        if max_windows is not None and window_index >= max_windows:
            break

        window_tokens = tokens[start : start + window_size]
        if len(window_tokens) < 2:
            break

        pad = False
        if len(window_tokens) < window_size:
            match final_window:
                case "drop":
                    break
                case "error":
                    raise ValueError(f"Final window has {len(window_tokens)} tokens; expected {window_size}.")
                case "pad":
                    pad = True
                case _:
                    raise ValueError(f"Unknown final_window policy: {final_window}")

        batch = tokens_to_lm_batch(
            window_tokens,
            seq_len=seq_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            pad=pad,
        )
        valid_targets = int(np.asarray(batch.loss_masks).sum())
        if valid_targets > 0:
            yield TokenWindow(index=window_index, tokens=window_tokens, batch=batch, valid_targets=valid_targets)
            window_index += 1
