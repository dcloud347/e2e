"""CLI for text generation with inner-loop TTT adaptation."""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from ttt.inference.runtime import adapt_text_batch, compose_text_eval_config, load_text_eval_runtime, next_token_logits, summarize_config
from ttt.inference.tokenization import encode_text, load_tokenizer, tokens_to_prompt_batch, truncate_tokens, validate_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw text with a TTT-E2E checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Orbax checkpoint directory, local path or gs:// URI.")
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B", help="Hugging Face tokenizer id or local tokenizer directory.")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token. Defaults to HF_TOKEN.")
    parser.add_argument("--prompt", help="Raw input prompt. Mutually exclusive with --prompt-file.")
    parser.add_argument("--prompt-file", help="Path to a UTF-8 prompt file. Mutually exclusive with --prompt.")
    parser.add_argument("--experiment", default="1b/pretrain/pretrain-1b-e2e", help="Hydra experiment preset.")
    parser.add_argument("--deploy", default="interactive", help="Hydra deploy preset.")
    parser.add_argument("--seq-length", type=int, help="Override training.seq_length. Defaults to the experiment config.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0 means greedy decoding.")
    parser.add_argument("--top-k", type=int, help="Keep only the top-k logits before sampling.")
    parser.add_argument("--top-p", type=float, help="Keep the smallest probability mass >= top-p before sampling.")
    parser.add_argument("--seed", type=int, default=0, help="NumPy sampling seed.")
    parser.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=True, help="Prepend the model BOS token before tokenization output.")
    parser.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=False, help="Append the model EOS token after tokenization output.")
    parser.add_argument("--stop-on-eos", action=argparse.BooleanOptionalAction, default=True, help="Stop generation after the model emits EOS.")
    parser.add_argument("--skip-special-tokens", action=argparse.BooleanOptionalAction, default=True, help="Skip special tokens when decoding output.")
    parser.add_argument("--strict-tokenizer", action=argparse.BooleanOptionalAction, default=True, help="Fail on tokenizer/model special-token mismatch.")
    parser.add_argument("--adaptation-only", action="store_true", help="Run prompt adaptation, print prompt stats, and exit without decoding.")
    parser.add_argument("--output-token-ids", action="store_true", help="Print prompt and generated token ids.")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override. Can be passed multiple times.")
    parser.add_argument("--print-config", action="store_true", help="Print the compact runtime config summary before loading the model.")
    return parser.parse_args()


def _read_prompt(*, prompt: str | None, prompt_file: str | None) -> str:
    if (prompt is None) == (prompt_file is None):
        raise ValueError("Pass exactly one of `--prompt` or `--prompt-file`.")
    if prompt is not None:
        return prompt
    with open(prompt_file, encoding="utf-8") as f:
        return f.read()


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if value < 1:
        raise ValueError(f"value must be positive, got {value}.")
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}.")
    return ((value + multiple - 1) // multiple) * multiple


def _sequence_length_for_tokens(num_tokens: int, *, max_length: int, multiple: int) -> int:
    if max_length % multiple != 0:
        raise ValueError(f"Maximum sequence length {max_length} must be divisible by {multiple}.")
    return min(max_length, _ceil_to_multiple(num_tokens, multiple))


def _device_put_batch(batch, mesh):
    sharding = jax.NamedSharding(mesh, P())
    return jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), sharding) if x is not None else None, batch)


def _device_put_scalar(value: int, mesh):
    sharding = jax.NamedSharding(mesh, P())
    return jax.device_put(jnp.asarray(value, dtype=jnp.int32), sharding)


def _filter_top_k(logits: np.ndarray, top_k: int | None) -> np.ndarray:
    if top_k is None:
        return logits
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if top_k >= logits.shape[-1]:
        return logits

    keep_indices = np.argpartition(logits, -top_k)[-top_k:]
    filtered = np.full_like(logits, -np.inf)
    filtered[keep_indices] = logits[keep_indices]
    return filtered


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def _filter_top_p(probs: np.ndarray, top_p: float | None) -> np.ndarray:
    if top_p is None:
        return probs
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if top_p == 1.0:
        return probs

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)
    keep_sorted = (cumulative - sorted_probs) < top_p
    keep_sorted[0] = True

    filtered = np.zeros_like(probs)
    kept_indices = sorted_indices[keep_sorted]
    filtered[kept_indices] = probs[kept_indices]
    total = filtered.sum()
    if total <= 0.0:
        raise ValueError("top_p filtering removed all probability mass.")
    return filtered / total


def sample_next_token(
    logits: np.ndarray,
    *,
    rng: np.random.Generator,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Select a token id from a logits vector."""

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {logits.shape}.")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}.")
    if temperature == 0.0:
        return int(np.argmax(logits))

    logits = _filter_top_k(logits / temperature, top_k)
    probs = _softmax(logits)
    probs = _filter_top_p(probs, top_p)
    return int(rng.choice(logits.shape[-1], p=probs))


def _build_prompt_batch(tokens: np.ndarray, *, cfg, tokenizer, multiple: int):
    context_tokens = truncate_tokens(tokens, max_length=cfg.training.seq_length)
    if len(context_tokens) == 0:
        raise ValueError("Prompt produced no tokens. Use --add-bos or provide non-empty text.")

    seq_len = _sequence_length_for_tokens(len(context_tokens), max_length=cfg.training.seq_length, multiple=multiple)
    batch = tokens_to_prompt_batch(
        context_tokens,
        eos_token_id=cfg.model.eos_token_id,
        seq_len=seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    return context_tokens, batch, len(context_tokens) - 1


def _adapt_prompt(runtime, batch):
    batch = _device_put_batch(batch, runtime.mesh)
    with runtime.mesh:
        return adapt_text_batch(runtime.model, runtime.state, batch)


def _next_logits_for_context(runtime, adapted_model, tokens: np.ndarray, tokenizer) -> np.ndarray:
    context_tokens, batch, last_index = _build_prompt_batch(tokens, cfg=runtime.cfg, tokenizer=tokenizer, multiple=runtime.cfg.model.mini_batch_size)
    batch = _device_put_batch(batch, runtime.mesh)
    last_index = _device_put_scalar(last_index, runtime.mesh)
    with runtime.mesh:
        logits = next_token_logits(adapted_model, runtime.state, batch, last_index)
    return np.asarray(jax.device_get(logits)), context_tokens


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {args.max_new_tokens}.")
    if args.temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {args.temperature}.")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError(f"top_k must be positive, got {args.top_k}.")
    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {args.top_p}.")

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

    prompt = _read_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    prompt_tokens = encode_text(
        tokenizer,
        prompt,
        bos_token_id=cfg.model.bos_token_id,
        eos_token_id=cfg.model.eos_token_id,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
    )

    adapt_multiple = cfg.model.mini_batch_size * max(cfg.training.inner_remat_freq, 1)
    context_tokens, prompt_batch, _last_index = _build_prompt_batch(prompt_tokens, cfg=cfg, tokenizer=tokenizer, multiple=adapt_multiple)

    print(f"prompt_tokens: {len(prompt_tokens)}")
    print(f"adaptation_tokens: {len(context_tokens)}")
    print(f"adaptation_seq_length: {len(prompt_batch.input_ids)}")
    if len(context_tokens) < len(prompt_tokens):
        print(f"warning: prompt was left-truncated to {len(context_tokens)} tokens")

    runtime = load_text_eval_runtime(cfg)
    adapt_result = _adapt_prompt(runtime, prompt_batch)

    prompt_loss = np.asarray(jax.device_get(adapt_result.loss))
    print(f"prompt_loss: {float(prompt_loss):.6f}")

    if args.output_token_ids:
        print(f"prompt_token_ids: {prompt_tokens.tolist()}")

    if args.adaptation_only or args.max_new_tokens == 0:
        print("completion:")
        return

    rng = np.random.default_rng(args.seed)
    generated_tokens: list[int] = []
    decode_tokens = np.asarray(context_tokens, dtype=np.int32)

    for _ in range(args.max_new_tokens):
        logits, decode_context = _next_logits_for_context(runtime, adapt_result.adapted_model, decode_tokens, tokenizer)
        next_token = sample_next_token(logits, rng=rng, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        generated_tokens.append(next_token)
        decode_tokens = np.concatenate([decode_context, np.asarray([next_token], dtype=np.int32)])

        if args.stop_on_eos and next_token == cfg.model.eos_token_id:
            break

    completion = tokenizer.decode(generated_tokens, skip_special_tokens=args.skip_special_tokens)

    print("prompt:")
    print(prompt)
    print()
    print("completion:")
    print(completion)
    if args.output_token_ids:
        print(f"generated_token_ids: {generated_tokens}")


if __name__ == "__main__":
    main()
