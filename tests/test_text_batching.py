import numpy as np
import pytest

from ttt.inference.tokenization import iter_token_windows, tokens_to_lm_batch, tokens_to_prompt_batch, truncate_tokens


def test_tokens_to_lm_batch_shifts_targets():
    batch = tokens_to_lm_batch(
        np.asarray([128000, 10, 11, 12, 128001], dtype=np.int32),
        seq_len=4,
        bos_token_id=128000,
        eos_token_id=128001,
    )

    np.testing.assert_array_equal(batch.input_ids, np.asarray([128000, 10, 11, 12], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_tokens, np.asarray([10, 11, 12, 128001], dtype=np.int32))
    np.testing.assert_array_equal(batch.loss_masks, np.asarray([True, True, True, True]))


def test_tokens_to_lm_batch_masks_padding_targets():
    batch = tokens_to_lm_batch(
        np.asarray([128000, 10, 11], dtype=np.int32),
        seq_len=4,
        bos_token_id=128000,
        eos_token_id=128001,
        pad=True,
    )

    np.testing.assert_array_equal(batch.input_ids, np.asarray([128000, 10, 11, 128001], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_tokens, np.asarray([10, 11, 128001, 128001], dtype=np.int32))
    np.testing.assert_array_equal(batch.loss_masks, np.asarray([True, True, False, False]))


def test_tokens_to_lm_batch_strict_short_window_errors():
    with pytest.raises(ValueError, match="Need 5 tokens"):
        tokens_to_lm_batch(
            np.asarray([128000, 10, 11], dtype=np.int32),
            seq_len=4,
            bos_token_id=128000,
            eos_token_id=128001,
        )


def test_iter_token_windows_drops_short_final_window():
    windows = list(
        iter_token_windows(
            np.asarray([128000, 10, 11, 12, 13, 14, 15], dtype=np.int32),
            seq_len=4,
            bos_token_id=128000,
            eos_token_id=128001,
            final_window="drop",
        )
    )

    assert len(windows) == 1
    assert windows[0].valid_targets == 4


def test_iter_token_windows_pads_short_final_window():
    windows = list(
        iter_token_windows(
            np.asarray([128000, 10, 11, 12, 13, 14, 15], dtype=np.int32),
            seq_len=4,
            bos_token_id=128000,
            eos_token_id=128001,
            final_window="pad",
        )
    )

    assert len(windows) == 2
    assert windows[0].valid_targets == 4
    assert windows[1].valid_targets == 2


def test_truncate_tokens_keeps_rightmost_tokens():
    tokens = truncate_tokens(np.asarray([1, 2, 3, 4, 5], dtype=np.int32), max_length=3)

    np.testing.assert_array_equal(tokens, np.asarray([3, 4, 5], dtype=np.int32))


def test_tokens_to_prompt_batch_masks_last_real_token():
    batch = tokens_to_prompt_batch(
        np.asarray([128000, 10, 11], dtype=np.int32),
        eos_token_id=128001,
    )

    np.testing.assert_array_equal(batch.input_ids, np.asarray([128000, 10, 11], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_tokens, np.asarray([10, 11, 128001], dtype=np.int32))
    np.testing.assert_array_equal(batch.loss_masks, np.asarray([True, True, False]))


def test_tokens_to_prompt_batch_pads_to_seq_len():
    batch = tokens_to_prompt_batch(
        np.asarray([128000, 10, 11], dtype=np.int32),
        eos_token_id=128001,
        seq_len=5,
        pad_token_id=0,
    )

    np.testing.assert_array_equal(batch.input_ids, np.asarray([128000, 10, 11, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(batch.target_tokens, np.asarray([10, 11, 128001, 0, 0], dtype=np.int32))
    np.testing.assert_array_equal(batch.loss_masks, np.asarray([True, True, False, False, False]))
