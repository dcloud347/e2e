"""语言模型数据加载。

数据集要求是本地 Zarr array，通常包含 `train`、`val` 等 split。
每个样本会读取连续的 `seq_len + 1` 个 token，并转换成 input/target 错位一位的 Batch。
"""

import grain.python as grain
import jax
import numpy as np
import zarr.codecs
import zarr.storage

from ttt.model.data import Batch


class Dataset(grain.RandomAccessDataSource):
    """从 Zarr split 中按固定长度随机访问 token 序列。"""

    def __init__(self, *, path: str, split: str, seq_len: int):
        # 数据预期使用 zstd 压缩；这里的 codec 需要和写入数据时的格式一致。
        codec = zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle)

        store = zarr.storage.LocalStore(path, read_only=True)

        # split 会映射到 Zarr 里的 /train、/val 等数组。
        self._dataset = zarr.open_array(store, path=f"/{split}", codec=codec)

        self.split = self._dataset
        self.seq_len = seq_len

    def __getitem__(self, idx):
        # 取 seq_len + 1 个 token，这样可以构造 seq_len 个 next-token 预测目标。
        sample = self.split[idx * self.seq_len : (idx + 1) * self.seq_len + 1]
        assert len(sample) == (self.seq_len + 1), "Loader got a sequence with the wrong length!"
        return sample

    def __len__(self):
        # 最后不足一个完整序列的 token 会被丢弃。
        return (self.split.shape[0] - 1) // self.seq_len


class DummyDataset(grain.RandomAccessDataSource):
    """调试用的随机 token 数据集，不需要真实数据文件。"""

    def __init__(self, *, seq_len: int, num_tokens: int = 2**25):
        self.seq_len = seq_len
        self.num_tokens = num_tokens

    def __getitem__(self, idx):
        # idx 不参与随机数种子；这里仅用于快速跑通 shape 和训练循环。
        sample = np.random.randint(0, 20, (self.seq_len + 1,), dtype=np.int32)
        return sample

    def __len__(self):
        return (self.num_tokens - self.seq_len - 1) // self.seq_len


def _to_batch(
    data: np.ndarray,
    *,
    bos_token_id: int,
    eos_token_id: int,
) -> Batch:
    """把一段 token 转成 causal language modeling 的 Batch。"""

    tokens = np.asarray(data)
    return Batch(
        # 输入是前 seq_len 个 token，目标是向右错一位后的 token。
        input_ids=tokens[:-1],
        target_tokens=tokens[1:],
        # BOS 只表示序列开始，不应该作为预测目标计入 loss。
        loss_masks=(tokens[1:] != bos_token_id),
    )


def lm_dataset(
    *,
    path: str,
    split: str,
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    seed=None,
    repeat: bool,
    shard_index: int | None = None,
    shard_count: int | None = None,
    shuffle: bool = True,
) -> grain.MapDataset:
    """创建真实语言模型数据集。

    多机训练时，每个 JAX process 只读取自己的 shard，避免不同 host 重复消费同一批数据。
    返回值是 Grain MapDataset，后续在训练入口里转成 iterator。
    """

    if shard_index is None:
        shard_index = jax.process_index()
    if shard_count is None:
        shard_count = jax.process_count()

    assert global_batch_size % shard_count == 0
    host_batch_size = global_batch_size // shard_count

    source = Dataset(path=path, split=split, seq_len=seq_len)
    dataset = grain.MapDataset.source(source)

    if shuffle:
        # repeat 前先 shuffle，确保每轮看到的数据顺序不同。
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    dataset_length = len(source)

    if repeat:
        print(f"Repeating dataset. Length {dataset_length}.")
        dataset = dataset.repeat()
    else:
        # 评估时不 repeat，并把 batch 数裁到 process 数的整数倍，方便所有 host 同步结束。
        dataset_length = len(dataset)
        trimmed_length = (dataset_length // shard_count) * shard_count  # Drop remainder
        dataset = dataset[:trimmed_length]
        print(f"Trimming dataset. Initial length {dataset_length}. New length {trimmed_length}.")

    # 每个 process 只取属于自己的数据片段。
    dataset = dataset[shard_index::shard_count]

    return dataset


def dummy_dataset(
    seq_len: int,
    global_batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    repeat: bool = False,
    num_tokens: int = 2**25,
):
    """创建随机数据集，用于无真实数据时测试训练流程。"""

    shard_index = jax.process_index()
    shard_count = jax.process_count()

    dataset = grain.MapDataset.source(
        DummyDataset(seq_len=seq_len, num_tokens=num_tokens),
    )

    host_batch_size = global_batch_size // shard_count
    dataset = dataset.map(
        lambda data: _to_batch(
            data,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
    ).batch(batch_size=host_batch_size, drop_remainder=True)

    if repeat:
        print("Repeating dataset.")
        dataset = dataset.repeat()

    # 和真实数据集保持同样的多进程切分方式。
    dataset = dataset[shard_index::shard_count]
    return dataset
