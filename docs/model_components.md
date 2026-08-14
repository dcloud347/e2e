# ttt/model 组件设计说明

这份文档按代码里的真实调用链来读：先看数据容器，再看注意力和 Block，最后看 `TransformerModel`、`CausalLM` 和 `MetaModel`。  
如果只想抓主线，可以先看“总流程”和“MetaModel”两节。

## 代码地图

| 文件 | 主要组件 | 作用 |
| --- | --- | --- |
| [`ttt/model/data.py`](../ttt/model/data.py) | `Batch`, `BaseModelOutput` | 输入和输出的 pytree 容器 |
| [`ttt/model/loss.py`](../ttt/model/loss.py) | `cross_entropy_loss_and_accuracy`, `token_log_probs` | next-token loss 和日志指标 |
| [`ttt/model/attention.py`](../ttt/model/attention.py) | `NormalLinear`, `AttentionBase`, `Attention`, `SWAFull`, `SWA` | RoPE + 多种序列建模层 |
| [`ttt/model/transformer.py`](../ttt/model/transformer.py) | `SwiGLUMLP`, `Block`, `BlockCollection`, `BlockCollectionSplit`, `TransformerModel`, `CausalLM`, `MetaModel` | 主模型和 E2E TTT 逻辑 |
| [`ttt/model/sharding.py`](../ttt/model/sharding.py) | `ModelSharding` | 参数和 batch 的 mesh 切分 |
| [`ttt/model/loop.py`](../ttt/model/loop.py) | `train_on_sequence`, `Evaluator` | 训练/评估外层循环 |

## 总流程

```text
Batch(input_ids, target_tokens, loss_masks)
  -> token embedding
  -> block stack
  -> final RMSNorm
  -> lm_head / tied embedding
  -> logits
  -> cross entropy loss
```

E2E TTT 的关键区别在于：

```text
长序列
  -> 先跑 prefix blocks 一次
  -> 再按 mini_batch_size 切成 chunk
  -> 每个 chunk 只跑 suffix blocks
  -> 内循环只更新 suffix 里的 prime FFN 参数
```

也就是说，模型不是“整段 32K/128K 一口吞”，而是“prefix 先读上下文，suffix 边读边适配”。

## 1. 数据容器

### `Batch`

`Batch` 是语言模型的输入包装，字段如下：

- `input_ids`: 输入 token
- `target_tokens`: 右移一位后的预测目标
- `loss_masks`: 哪些位置参与 loss
- `attention_mask`: 预留字段，当前主路径主要靠 causal / window mask
- `position_ids`: 可选位置 id
- `index`: 静态调试字段，记录切片位置

它被做成 `eqx.Module`，原因很直接：JAX 需要它能像 pytree 一样被 `jit`、`vmap`、`scan` 传递。  
`slice_index()` 会把 pytree 每个 leaf 同步切片，并保留 `index`，方便在 chunk 级别调试。

### `BaseModelOutput`

这是模型前向的统一返回值：

- `state`: Equinox state
- `last_hidden_state`: 最后一层 hidden state
- `logits`: vocab logits

它的意义是把“张量输出”和“可变状态”绑在一起，避免长序列里 KV cache 或 step state 被丢掉。

## 2. 线性层和 RoPE

### `NormalLinear`

这是项目自定义的无 bias 线性层：

- 权重用正态分布初始化
- 参数 dtype 和计算 dtype 分开
- 前向时会先 `promote_dtype()`，避免 fp32 参数和 bf16/fp16 计算混在一起
- `name` 只用于 checkpoint / remat 打标，便于调试

它是整个模型里所有投影层的统一基元：Q/K/V/O、MLP 的 `w1/w2/w3`、输出头都通过它构建。

### `precompute_freqs_cis()` 和 `apply_rotary_emb()`

RoPE 的实现拆成两步：

1. 预先生成复数形式的频率表
2. 在 query / key 上做复数旋转

这里把频率表预留到 `2 * seq_len`，是为了兼容 sliding-window 场景里“历史窗口 + 当前 chunk”的位置索引。  
实现上直接用复数乘法，比较贴近 RoPE 的数学形式，也便于 JAX 编译。

## 3. 注意力族

### `AttentionBase`

`AttentionBase` 把所有 attention 共享的部分抽出来：

- `wq/wk/wv/wo` 四个投影
- `q_norm` / `k_norm`
- head split / merge
- RoPE 应用
- `core_attention_op()` 这一层真正执行 dot-product attention

它的设计重点有三个：

- `qk_norm`：先对 Q/K 做 RMSNorm，长上下文训练更稳
- `freqs_cis`：按需生成 RoPE 频率表
- `force_flash`：需要 cuDNN 路径时加 sharding constraint，减少布局冲突

`core_attention_op()` 里直接调用 `jax.nn.dot_product_attention()`，当前不支持 attention dropout。

### `Attention`

这是标准 causal self-attention：

- 输入整段 chunk
- 用 `is_causal=True`
- 适合普通 pretrain / baseline 配置

当 `force_flash=True` 或处于 prefix 路径时，代码会给 Q/K/V 和输出加额外的 sharding 约束，尽量走 cuDNN 实现。

### `SWAFull`

这是“原生 local window attention”版本：

- 仍然处理整段 chunk
- 但 `local_window_size=(window_size - 1, 0)`
- 适合需要直接用 cuDNN local-window kernel 的路径

它和 `Attention` 的差别在于注意力范围，而不是模型外壳。

### `SWA`

这是手动 KV cache 的 sliding-window attention，是真正为长上下文 chunk 化设计的版本。

核心状态有两个：

- `kv_cache_index`: 最近窗口的 K/V cache
- `chunk_index`: 当前是第几个 chunk

它的执行逻辑可以理解成：

1. 计算当前 chunk 的 Q/K/V
2. 取出上一个 chunk 留下的窗口缓存
3. 把缓存和当前 chunk 拼起来
4. 只保留最后 `window_size` 个 K/V 作为下一步缓存
5. 用显式 causal mask 做 attention
6. 把新 cache 写回 `nn.State`

prefix 路径会走 `full_sw_attention()`，也就是用 local-window attention 一次性算完整段，但不更新 cache。  
这保证 prefix 和 suffix 的语义一致，只是 prefix 不需要为后续 chunk 维护状态。

## 4. FFN 和 Block

### `SwiGLUMLP`

这是标准的 SwiGLU MLP：

```text
w1(x) --SiLU--\
               * -> w2 -> dropout
w3(x) ---------/
```

设计上很朴素：

- `w1` 和 `w3` 做门控
- `w2` 投回 hidden size
- 用 residual dropout

### `PrimeStorage`

这是 E2E TTT 额外引入的“prime 参数仓库”。

它只在 `config.prime=True` 时存在，里面保存的是每个 suffix block 对应的一套：

- `ffn_prime_norm`
- `ffn_prime_post_norm`
- `feed_forward_prime`

这部分参数和普通 FFN 分开存，目的是让内循环只更新 prime FFN，而不碰底层主干权重。  
当前实现只支持 `feed_forward_prime="swiglu"`。

### `Block`

一个 `Block` 由四层逻辑组成：

1. 序列建模层：`Attention` / `SWA` / `SWAFull`
2. 普通 FFN：`SwiGLUMLP`
3. 可选 prime FFN：只给 suffix blocks 用
4. pre-norm / post-norm + residual

默认配置里 `pre_norm` 和 `post_norm` 都可以打开，所以代码不是硬编码成单一 Transformer 变体，而是把规范化放成开关。

大致数据流是：

```text
x
  -> (pre norm)
  -> attention / SWA
  -> (post norm)
  -> residual add
  -> (optional prime FFN)
  -> regular FFN
  -> residual add
```

如果有 `feed_forward_prime`，suffix block 会先走 prime FFN，再走普通 FFN。  
这就是 E2E TTT 里“可快速适配”的位置。

### `BlockCollection`

这是完整的 block stack：

- 用 `jax.vmap` 一次性初始化所有层
- 用 `scan_or_loop()` 顺序执行每一层
- 从 `state.substate(self.blocks)` 里取出属于 block stack 的子状态

这里的状态主要是像 SWA cache 这种“按层保存”的东西。  
`scan_or_loop()` 允许在 `scan` 和 Python loop 之间切换，前者更快，后者更适合排查数值问题。

### `BlockCollectionSplit`

这是 E2E TTT 的核心结构之一。

它把完整 block stack 拆成两段：

- `prefix_blocks`: 前面一段，先读完整上下文
- `suffix_blocks`: 后面一段，按 chunk 运行并参与内循环

它还会同步拆分 `nn.State`：

- prefix state
- suffix state

如果某些 state 长度不够，还会补零，确保 suffix scan 的形状对齐。

为什么要这么拆：

- prefix 只负责“阅读上下文”
- suffix 负责“边读边适配”
- 内循环只需要盯住 suffix 里的 prime 参数

这就是长上下文 test-time training 的核心工程化表达。

## 5. Transformer 主体和 LM Head

### `TransformerModel`

它是“不带语言模型输出头”的主干：

- `wte`: token embedding
- `dropout`
- `h`: block stack
- `ln_f`: final RMSNorm

它提供三种入口：

- `wte_call()`: 只做 embedding
- `prefix_call()`: 走拆分后的 prefix blocks
- `suffix_call()`: 走拆分后的 suffix blocks，再接 final RMSNorm

普通前向则是 `embedding -> blocks -> ln_f`。

### `CausalLM`

`CausalLM` 在 `TransformerModel` 外面包一层 vocab head。

两种输出方式：

- `tie_word_embeddings=True`：直接复用 `wte.weight.T`
- 否则：单独初始化 `lm_head`

这也是配置里为什么会出现 `tie_word_embeddings` 的原因。  
对于这个项目来说，词嵌入共享是默认推荐路径。

## 6. MetaModel

`MetaModel` 是训练层真正接触的高层模型。

它额外持有：

- `config`
- dtype 信息
- `step_index`
- `language_model`

### `get_ilr_multiplier()` 和 `inner_optimizer()`

内循环学习率不是固定常数，而是可以做 warmup 的。

如果 `ilr_warmup_steps > 0`，它会把 `ilr_init` 线性 warm 到 `optimizer_inner.lr`。  
然后 `inner_optimizer()` 用这个倍率动态构造内循环优化器。

### `lm_loss()`

这是单段 token 的 next-token loss：

- 没传 `prefix_outputs` 时，跑完整 `CausalLM`
- 传了 `prefix_outputs` 时，只跑 suffix

这正好匹配 E2E TTT 的两种路径：

- 预训练：整段普通 LM
- meta：prefix 先算好，suffix 边跑边更新

### `inner_loop_step()`

这一步是内循环更新的最小单元：

1. 对当前 chunk 算 `lm_loss`
2. 反传得到梯度
3. 只保留 `spec_inner` 指定的参数梯度
4. 用内循环优化器更新这些参数
5. 返回新的模型、opt state、state 和指标

也就是说，内循环不是“全模型都学”，而是只学被规则选中的那部分，通常就是 suffix block 里的 prime FFN。

### `loss_for_sequence()`

这是最重要的入口。

它会先：

- 把完整 `BlockCollection` 拆成 prefix/suffix
- 把 `nn.State` 也拆成 prefix/suffix
- 克隆当前模型，避免原对象被就地污染

然后根据训练模式分两条路：

#### `train_mode = meta`

- 把模型 cast 到 `state_dtype`
- 创建内循环优化器状态
- prefix 对整条序列只算一次
- 把长序列切成 `mini_batch_size` chunk
- 每个 chunk 执行一次 `inner_loop_step()`
- `scan_remat_chunk()` 负责把 chunk 循环做成可 remat 的扫描

#### `train_mode = pretrain`

- 不做内循环
- 直接按 chunk 顺序算普通 LM loss

最后它会把 metrics 展平，方便后面的日志记录。

### 参数选择

`MetaModel` 里有两个很关键的接口：

- `trainable_parameters()`
- `inner_parameters()`

它们都通过 `training.spec_outer` / `training.spec_inner` 做路径匹配。  
匹配规则在 [`ttt/utils/filter_utils.py`](../ttt/utils/filter_utils.py) 里，语法类似文件路径：

- `.` 分隔层级
- `*` 匹配一层
- `**` 匹配任意深度
- `exclude ...` 代表排除

典型 E2E 配置是：

```yaml
spec_inner: ["language_model.**.suffix_blocks.feed_forward_prime.**"]
```

这表示内循环只更新 suffix blocks 里的 prime FFN。

## 7. Loss

[`ttt/model/loss.py`](../ttt/model/loss.py) 里只有两件事：

- `cross_entropy_loss_and_accuracy()`
- `token_log_probs()`

### `cross_entropy_loss_and_accuracy()`

它计算 masked next-token cross entropy：

- `loss_masks` 为 0 的 token 不参与 loss
- 先按每条序列内部做平均
- 再对 batch 求平均

函数名里保留了 `accuracy`，但当前第二个返回值其实也是 loss，主要用于日志。

### `token_log_probs()`

返回每个目标 token 的 log probability，方便记录 token-level NLL 曲线。

## 8. 参数 sharding

[`ttt/model/sharding.py`](../ttt/model/sharding.py) 把设备 mesh 分成两个轴：

- `data`: 数据并行，只复制模型，切 batch
- `state`: 模型/参数并行，切大矩阵

这里的 `state` 是 mesh 轴名字，不是 `nn.State`，两者不要混。

### `ModelSharding`

它会根据总设备数和 `n_state_parallel` 自动推导 `n_data_parallel`，然后创建：

```text
mesh = (data, state)
```

`shard_params()` 再按参数形状分配 `PartitionSpec`：

- `ln_f` 这类一维参数：`P("state")`
- `wte`、block norm、`lm_head`：`P(None, "state")`
- `wq/wk/wv`、`w1/w3`：`P(None, "state", None)`
- `wo`、`w2`：`P(None, None, "state")`

prime 参数沿用和普通 FFN 一样的切法。  
这样做的好处是：分片方向和 matmul 的主计算维度对齐，模型更容易在多卡上跑得动。

## 9. 典型配置怎么对应到模型

你会在实验配置里看到两种常见模式：

### FA baseline

```yaml
seq_modeling_block: self_attention
force_flash: True
train_mode: pretrain
```

这对应的是标准 Transformer + flash attention，不走内循环。

### E2E TTT

```yaml
seq_modeling_block: SWA
prime: True
suffix_len: 3   # 125M / 125M extension
spec_inner: ["language_model.**.suffix_blocks.feed_forward_prime.**"]
train_mode: meta
```

这对应的是：

- 序列建模层用 sliding-window attention
- suffix block 带 prime FFN
- 内循环只更新 prime FFN
- prefix 先读完整上下文，suffix 分 chunk 自适应

## 10. 建议的阅读顺序

如果你想快速吃透实现，建议按这个顺序看源码：

1. [`ttt/model/data.py`](../ttt/model/data.py)
2. [`ttt/model/attention.py`](../ttt/model/attention.py)
3. [`ttt/model/transformer.py`](../ttt/model/transformer.py)
4. [`ttt/model/loss.py`](../ttt/model/loss.py)
5. [`ttt/model/sharding.py`](../ttt/model/sharding.py)
6. [`ttt/model/loop.py`](../ttt/model/loop.py)

这样你会先搞懂数据怎么流，再看参数怎么更新，最后看多卡怎么切。
