"""CLI for raw text loss evaluation with inner-loop TTT."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from ttt.inference.runtime import compose_text_eval_config, eval_text_batch, load_text_eval_runtime, summarize_config
from ttt.inference.tokenization import encode_text, iter_token_windows, load_tokenizer, read_text_input, validate_tokenizer
from ttt.model.transformer import MetaModel


@dataclass
class WindowEval:
    """Metrics for one evaluated text window."""

    index: int
    tokens: int
    valid_targets: int
    loss: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw text with the TTT-E2E inner loop.")
    parser.add_argument("--checkpoint", required=True, help="Orbax checkpoint directory, local path or gs:// URI.")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B", help="Hugging Face tokenizer id or local tokenizer directory.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--text", help="Raw input text. Mutually exclusive with --text-file.")
    parser.add_argument("--text-file", help="Path to a UTF-8 text file. Mutually exclusive with --text.")
    parser.add_argument("--experiment", default="1b/pretrain/pretrain-1b-e2e", help="Hydra experiment preset.")
    parser.add_argument("--deploy", default="interactive", help="Hydra deploy preset.")
    parser.add_argument("--seq-length", type=int, help="Override training.seq_length. Defaults to the experiment config.")
    parser.add_argument("--max-windows", type=int, help="Maximum number of token windows to evaluate.")
    parser.add_argument("--final-window", choices=["pad", "drop", "error"], default="pad", help="How to handle a short final token window.")
    parser.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=True, help="Prepend the model BOS token before tokenization output.")
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=False, help="Append the model EOS token after tokenization output.")
    parser.add_argument("--strict-tokenizer", action=argparse.BooleanOptionalAction, default=True, help="Fail on tokenizer/model special-token mismatch.")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override. Can be passed multiple times.")
    parser.add_argument("--print-config", action="store_true", help="Print the compact runtime config summary before loading the model.")
    return parser.parse_args()


def _device_put_batch(batch, mesh):
    sharding = jax.NamedSharding(mesh, P())
    return jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), sharding) if x is not None else None, batch)


def _eval_window(runtime, window) -> WindowEval:
    batch = _device_put_batch(window.batch, runtime.mesh)
    with runtime.mesh:
        _loss, metrics = eval_text_batch(runtime.model, runtime.state, batch)

    token_nll_loss = np.asarray(jax.device_get(metrics[MetaModel.MetricType.token_nll_loss]))
    loss_masks = np.asarray(window.batch.loss_masks, dtype=bool)
    total_nll = float(token_nll_loss[loss_masks].sum())
    valid_targets = int(loss_masks.sum())
    if valid_targets == 0:
        raise ValueError(f"Window {window.index} has no valid targets.")
    return WindowEval(index=window.index, tokens=len(window.tokens), valid_targets=valid_targets, loss=total_nll / valid_targets)


def main() -> None:
    args = parse_args()
    overrides = list(args.override)
    if args.seq_length is not None:
        overrides.append(f"training.seq_length={args.seq_length}")

    cfg = compose_text_eval_config(
        checkpoint=args.checkpoint,
        experiment=args.experiment,
        deploy=args.deploy,
        overrides=overrides,
    )

    if args.print_config:
        print(summarize_config(cfg))

    tokenizer = load_tokenizer(args.tokenizer, token=args.hf_token)
    warnings = validate_tokenizer(
        tokenizer,
        vocab_size=cfg.model.vocab_size,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        strict=args.strict_tokenizer,
    )
    for warning in warnings:
        print(f"warning: {warning}")

    text = read_text_input(text=args.text, text_file=args.text_file)
    tokens = encode_text(
        tokenizer,
        text,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    windows = list(
        iter_token_windows(
            tokens,
            seq_len=cfg.training.seq_length,
            bos_token_id=cfg.model.bos_token_id,
            eos_token_id=cfg.model.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            final_window=args.final_window,
            max_windows=args.max_windows,
        )
    )
    if not windows:
        raise ValueError("No token windows to evaluate. Use --final-window=pad or provide more text.")

    print(f"input_tokens: {len(tokens)}")
    print(f"windows: {len(windows)}")
    print(f"seq_length: {cfg.training.seq_length}")
    print(f"inner_chunks_per_window: {cfg.training.seq_length // cfg.model.mini_batch_size}")

    runtime = load_text_eval_runtime(cfg)

    results = [_eval_window(runtime, window) for window in windows]
    total_nll = sum(result.loss * result.valid_targets for result in results)
    total_targets = sum(result.valid_targets for result in results)
    aggregate_loss = total_nll / total_targets

    for result in results:
        print(f"window[{result.index}]: tokens={result.tokens} valid_targets={result.valid_targets} loss={result.loss:.6f} ppl={math.exp(result.loss):.6f}")

    print(f"valid_targets: {total_targets}")
    print(f"loss: {aggregate_loss:.6f}")
    print(f"perplexity: {math.exp(aggregate_loss):.6f}")


if __name__ == "__main__":
    main()

