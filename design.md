# Design: ttt-generate for NLP Output

## Goal

Add a generation path that turns the existing 1B/125M/3B TTT-E2E checkpoints into a text-in/text-out model.

The key behavior is:

1. Read a natural-language prompt.
2. Tokenize it with the Llama-3 tokenizer.
3. Run the existing inner-loop TTT adaptation on the prompt context.
4. Generate new tokens autoregressively from the adapted model.
5. Decode the generated tokens back to natural language.

This is a second-stage feature after `ttt-text-eval`. The model architecture stays unchanged.

## Non-Goals

- Do not introduce an encoder-decoder architecture.
- Do not add cross-attention over a separate encoder stream.
- Do not retrain the checkpoint from scratch.
- Do not change the tokenizer vocabulary.
- Do not change the existing Zarr training path.
- Do not implement serving infrastructure or streaming APIs in this pass.

## Current State

The repository already has:

- `ttt-text-eval` for raw text loss / perplexity.
- `MetaModel.loss_for_sequence(seq, state)` for inner-loop TTT over a single sequence.
- `Batch(input_ids, target_tokens, loss_masks)` as the LM input container.
- `CausalLM` and `TransformerModel` for token-level forward passes.

What is missing for generation:

- A way to obtain the final adapted model/state after the prompt inner loop.
- A reusable autoregressive decode loop.
- A raw-text CLI that prints decoded text instead of loss.

## Why generation needs extra work

`loss_for_sequence()` currently does two jobs at once:

1. perform prompt/context adaptation in meta mode;
2. compute and return aggregate loss metrics.

That is enough for eval, but not enough for decode because generation needs the final adapted parameters and state after the prompt has been consumed.

The first design change is to split adaptation from scoring.

## Proposed Model API

Add a new method to `MetaModel`:

```python
def adapt_on_sequence(self, seq: Batch, state: nn.State) -> AdaptResult:
    ...
```

Suggested return object:

```python
class AdaptResult(eqx.Module):
    adapted_model: MetaModel
    adapted_state: nn.State
    metrics: dict[MetaModel.MetricType, jnp.ndarray]
```

Behavior:

- `adapted_model` contains the prompt-conditioned inner-loop parameters.
- `adapted_state` contains the updated SWA / block state after consuming the prompt.
- `metrics` can include prompt loss, token NLL, and any adaptation statistics.

Then:

- `loss_for_sequence()` can call `adapt_on_sequence()` and keep its current loss behavior.
- `ttt-generate` can call `adapt_on_sequence()` and continue from the returned model/state.

## Generation Flow

### 1. Prompt tokenization

Input can be:

- `--prompt "..."`, or
- `--prompt-file path/to/file.txt`

Tokenization should use the same Llama-3 tokenizer rules as eval:

- optional BOS at the start
- optional EOS only if explicitly requested
- same `vocab_size`, `bos_token_id`, `eos_token_id`

For generation, the prompt is not converted into `target_tokens` for loss reporting unless we also want prompt perplexity.

### 2. Prompt adaptation

Run inner-loop TTT on the prompt context before generating.

Recommended behavior:

- use the prompt as the adaptation sequence;
- chunk it with the same `mini_batch_size`;
- update only `training.spec_inner`;
- preserve the final adapted model and state for decoding.

This keeps generation aligned with the existing meta-training objective:

```text
adapt on context -> decode next tokens
```

### 3. Autoregressive decoding

After adaptation, generate token by token.

At each step:

1. Feed the current token context through the adapted model.
2. Compute logits for the next token.
3. Sample or pick the next token.
4. Append it to the growing sequence.
5. Stop on EOS or `--max-new-tokens`.

The first implementation should be greedy or temperature-only decoding. Top-k and top-p can come later.

### 4. Decode to text

After token generation, use the tokenizer’s decode path to return natural language text.

CLI output should include:

- the original prompt
- the generated continuation
- optionally token ids for debugging

Example:

```text
prompt:
Explain test-time training in one paragraph.

completion:
Test-time training adapts the model on the current context ...
```

## Decoder Design

Add a small inference module:

```text
ttt/inference/generate.py
```

Responsibilities:

- parse CLI args
- load tokenizer
- load checkpoint
- adapt on the prompt
- autoregressively generate
- decode and print the final text

Suggested CLI:

```bash
ttt-generate \
  --checkpoint gs://ttt-e2e-checkpoints/1b_ttt_e2e_pretrain_dclm_8k_1x_cc \
  --tokenizer meta-llama/Meta-Llama-3-8B \
  --prompt "Explain test-time training in one paragraph." \
  --max-new-tokens 128
```

Optional flags:

- `--temperature`
- `--top-k`
- `--top-p`
- `--seed`
- `--prompt-file`
- `--output-token-ids`
- `--adaptation-only` for debugging

## Sequence Handling

The current model path assumes fixed-length windows for inner-loop work.

Generation needs two sequence modes:

### Prompt adaptation mode

Use the prompt as a fixed window or as multiple windows if it is longer than `seq_length`.

If the prompt exceeds the model context:

- either truncate from the left, or
- slide over windows and only keep the final adapted state.

Recommended first choice: truncate from the left to fit the maximum context, because this matches common causal LM prompting behavior and keeps the implementation simpler.

### Decode mode

Generation itself is incremental and does not require full fixed-length windows.

However, the model currently expects `Batch` objects and uses `seq.shape[0]` in attention code. For generation we need a careful implementation choice:

- either repeatedly call the model on the full growing prefix, or
- expose an incremental decode path that reuses the model state.

Recommended first implementation:

- keep the prompt adaptation path fixed-length;
- for decoding, call the model incrementally on the new token plus existing state if the attention/state path supports it;
- if incremental decoding is too intrusive, fall back to recomputing over the growing prefix for the first version and optimize later.

The fallback is slower, but it is the lowest-risk way to get correct text output first.

## State and Cache Management

This codebase uses `nn.State` heavily for SWA cache and other block state.

Generation must preserve:

- adapted inner-loop parameters;
- updated prefix/suffix state;
- chunk index / KV cache progression.

The safest design is to make the decode loop carry:

```python
carry = {
    "model": adapted_model,
    "state": adapted_state,
    "tokens": generated_tokens,
}
```

Each step returns a new carry and the sampled token.

## Resource Plan

Compared with `ttt-text-eval`, generation is a bit lighter on loss bookkeeping, but still needs the same model residency in memory.

Rough guidance:

- 125M: single A100 40GB should be fine.
- 1B: 1x H100 or 2x H100 with state parallel is safer.
- 3B: 4x to 8x H100 recommended.

The adapter step dominates the memory picture more than the generation loop itself.

## Implementation Phases

### Phase 1: Add `adapt_on_sequence`

- Factor prompt adaptation out of `loss_for_sequence()`.
- Return the final adapted model and state.
- Keep `loss_for_sequence()` behavior unchanged.

Exit criteria:

- Existing eval still passes.
- Generation code can access adapted model/state.

### Phase 2: Add greedy generation CLI

- Add `ttt-generate`.
- Tokenize prompt.
- Adapt on prompt.
- Decode `max_new_tokens` greedily.
- Decode output text.

Exit criteria:

- A prompt produces readable natural-language continuation.

### Phase 3: Add sampling controls

- Temperature
- Top-k
- Top-p
- Seeded sampling

Exit criteria:

- Output can be tuned between deterministic and stochastic decoding.

### Phase 4: Improve long-prompt handling

- Prompt truncation policy
- Sliding prompt windows
- Optional prompt perplexity

Exit criteria:

- Long prompts behave predictably and do not blow context limits.

## Testing Plan

### Unit tests

Add tests for:

- `adapt_on_sequence()` returns an adapted model/state pair.
- Prompt tokenization and special-token handling.
- Greedy token append logic.
- EOS stopping.
- Decode round-trip for a small tokenizer fixture if possible.

### Smoke tests

- Use a short prompt and `--max-new-tokens 4`.
- Verify the CLI prints text, not loss.
- Verify `ttt-generate` works with `--adaptation-only` or `--max-new-tokens 1` before trying longer runs.

## Risks

### The model is not a native decoder API

The current code was built around loss computation, not generation. The first generation version may need a conservative fallback that recomputes the full prefix rather than a fully incremental cache-aware decode path.

### Context length

Prompt adaptation can exceed the configured `seq_length`. We need a clear left-truncation or sliding-window policy.

### Tokenizer access

Llama-3 tokenizer access still depends on Hugging Face permissions or a local copy.

### State correctness

Because SWA maintains cache in `nn.State`, generation bugs can silently produce wrong text even when shapes look fine. Keep the first version simple and validate on tiny prompts.
