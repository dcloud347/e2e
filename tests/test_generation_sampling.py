import numpy as np
import pytest

from ttt.inference.generate import _sequence_length_for_tokens, sample_next_token


def test_sequence_length_for_tokens_rounds_to_multiple():
    assert _sequence_length_for_tokens(7, max_length=16, multiple=4) == 8
    assert _sequence_length_for_tokens(16, max_length=16, multiple=4) == 16


def test_sequence_length_for_tokens_requires_divisible_max_length():
    with pytest.raises(ValueError, match="must be divisible"):
        _sequence_length_for_tokens(7, max_length=15, multiple=4)


def test_sample_next_token_greedy_uses_argmax():
    token = sample_next_token(np.asarray([0.0, 3.0, 1.0]), rng=np.random.default_rng(0), temperature=0.0)

    assert token == 1


def test_sample_next_token_rejects_bad_temperature():
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(np.asarray([0.0, 1.0]), rng=np.random.default_rng(0), temperature=-1.0)


def test_sample_next_token_top_k_limits_candidates():
    rng = np.random.default_rng(0)
    samples = {sample_next_token(np.asarray([10.0, 9.0, -100.0]), rng=rng, temperature=1.0, top_k=2) for _ in range(20)}

    assert samples <= {0, 1}
