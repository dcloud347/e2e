"""按路径筛选模型参数的工具。

训练配置里的 `spec_outer` 和 `spec_inner` 会用这里的规则选择参数。
规则类似文件路径匹配：`.` 分隔层级，`*` 匹配一层，`**` 匹配任意多层，`exclude ...` 表示排除。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
from jax._src.lib import pytree
from jaxtyping import PyTree

if TYPE_CHECKING:
    from ttt.model.transformer import CausalLM, MetaModel


@dataclass
class SpecNode:
    """参数路径匹配规则中的一个节点。"""

    @staticmethod
    def from_string(string):
        """把配置字符串里的一个片段解析成具体节点。"""

        if string == "**":
            return DoubleWildNode()
        elif string == "*":
            return WildNode()
        elif string.isdigit():
            return IndexNode(int(string))
        else:
            return StringNode(string)

    @staticmethod
    def parse_spec_str(spec: str) -> list[SpecNode]:
        """把完整 spec 字符串按 `.` 拆成节点列表。"""

        return [SpecNode.from_string(string) for string in spec.split(".")]


@dataclass
class StringNode(SpecNode):
    """精确匹配属性名或字典 key。"""

    value: str


@dataclass
class WildNode(SpecNode):
    """匹配单层路径。"""

    pass


@dataclass
class DoubleWildNode(SpecNode):
    """匹配任意长度路径，可以出现在前缀或中间位置。"""

    pass


@dataclass
class IndexNode(SpecNode):
    """匹配 list/tuple/vmap 维度中的具体下标。"""

    index: int

    def __post_init__(self):
        assert self.index >= 0, "Negative indices are not allowed"


def matches(spec: list[SpecNode], path: list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey]) -> bool:
    """判断一条 pytree 路径是否匹配某个 spec。"""

    match spec, path:
        case [[], []]:
            # spec 和路径同时走完，说明完全匹配。
            return True
        case [[WildNode(), *s_rest], [_p_cur, *p_rest]]:
            return matches(s_rest, p_rest)
        case [[DoubleWildNode(), *s_rest], [_p_cur, *p_rest]]:
            # `**` 可以匹配 0 层，也可以继续吞掉当前路径节点。
            return matches(s_rest, p_rest) or matches(spec, p_rest)
        case [[IndexNode(n_i), *s_rest], [pytree.SequenceKey(s_i), *p_rest]]:
            return n_i == s_i and matches(s_rest, p_rest)
        case [[StringNode(n_s), *s_rest], [pytree.GetAttrKey(s_s) | pytree.DictKey(s_s), *p_rest]]:
            return n_s == s_s and matches(s_rest, p_rest)
        case _:
            return False


@dataclass
class Spec:
    """一条完整的 include/exclude 规则。"""

    is_exclude: bool
    spec_parts: list[SpecNode]

    @classmethod
    def from_string(cls, s: str):
        """解析配置里的 spec 字符串。"""

        is_exclude = False
        exclude_str = "exclude "
        if s.startswith(exclude_str):
            s = s[len(exclude_str) :]
            is_exclude = True

        return cls(is_exclude=is_exclude, spec_parts=SpecNode.parse_spec_str(s))


@dataclass
class SpecMatch:
    """某条规则在某个参数上的匹配结果。"""

    exclude: bool
    match: bool


def reduce_spec(spec_matches: list[SpecMatch]):
    """把多条 include/exclude 规则合并成最终是否选择该参数。"""

    current = False
    for spec_match in spec_matches:
        if spec_match.match:
            # 后面的规则可以覆盖前面的规则，exclude 会把当前参数移除。
            if not spec_match.exclude:
                current = True
            else:
                current = False

    return current


def get_filter_spec(tree: MetaModel | CausalLM, spec_strs: list[str], filter_type: str):
    """生成与模型同结构的布尔 pytree，True 表示该参数会被选中。"""

    specs = [Spec.from_string(spec_str) for spec_str in spec_strs]

    # 对每条 spec 分别计算它在每个 leaf 参数上的匹配结果。
    specs_matches = [
        jax.tree.map_with_path(lambda path, _value: SpecMatch(exclude=spec.is_exclude, match=matches(spec.spec_parts, path)), tree) for spec in specs
    ]

    for spec_str, spec_matches in zip(spec_strs, specs_matches):  # Every supplied spec must match at least one parameter
        if "index" not in spec_str:
            # 普通 spec 如果一个参数都没有匹配到，通常是配置写错了，直接报错。
            assert jax.tree.reduce(lambda a, b: a or b, jax.tree.map(lambda n: n.match, spec_matches)), (
                f"Spec {filter_type} did not match any parameters: {spec_str}"
            )

    # 多条规则按顺序合并，得到最终 mask。
    selected_params_filter_spec = jax.tree.map(lambda *entries: reduce_spec(entries), *specs_matches)

    return selected_params_filter_spec


def filter_parameters(tree: MetaModel | CausalLM, spec_strs: list[str], filter_type: str) -> MetaModel:
    """返回只保留匹配参数的 pytree，未匹配位置会变成 None。"""

    selected_params_filter_spec = get_filter_spec(tree, spec_strs, filter_type)

    selected_params = eqx.filter(tree, selected_params_filter_spec)

    return selected_params


## Helpers for printing the minimal paths of selected parameters. The implementations are only used in logging and can be ignored for functionality.


def _dict_flatten(d: dict) -> list[tuple[list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey], Any]]:
    """把嵌套字典展开成 `(path, value)` 列表。"""

    def flatten_gen(d):
        if isinstance(d, dict):
            for k, v in d.items():
                for path, value in flatten_gen(v):
                    yield [k, *path], value
        else:
            yield [], d

    return list(flatten_gen(d))


def _reduce_to_prefix_paths(tree: PyTree) -> list[tuple[list[pytree.GetAttrKey | pytree.SequenceKey | pytree.DictKey], Any]]:
    """把 pytree 压缩成最短的前缀路径表示，主要用于日志展示。"""

    def reduce_tree(tree):
        if not isinstance(tree, dict):
            return tree
        reduced = {k: reduce_tree(v) for k, v in tree.items()}
        assert len(reduced) > 0
        first = reduced[next(iter(reduced))]
        if not isinstance(first, dict) and all(v == first for v in reduced.values()):
            return first
        else:
            return reduced

    # 先把 pytree 变成以路径节点为 key 的嵌套字典。
    tree_from_path = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        tree_ptr = tree_from_path
        for p in path[:-1]:
            if p not in tree_ptr:
                tree_ptr[p] = {}
            tree_ptr = tree_ptr[p]
        tree_ptr[path[-1]] = value

    reduced_tree = reduce_tree(tree_from_path)
    return _dict_flatten(reduced_tree)


def filter_apply_updates(model, updates):
    """把 Optax updates 加到模型参数上；None 表示该位置不更新。"""

    model = jax.tree.map(lambda p, u: p + u if u is not None else p, model, updates)
    return model


def tree_path_to_string(path, sep=None):
    """把 JAX pytree path 转成字符串或字符串 tuple。"""

    keys = []
    for key in path:
        if isinstance(key, jax.tree_util.SequenceKey):
            keys.append(str(key.idx))
        elif isinstance(key, jax.tree_util.DictKey):
            keys.append(str(key.key))
        elif isinstance(key, jax.tree_util.GetAttrKey):
            keys.append(str(key.name))
        elif isinstance(key, jax.tree_util.FlattenedIndexKey):
            keys.append(str(key.key))
        else:
            keys.append(str(key))
    if sep is None:
        return tuple(keys)
    return sep.join(keys)


def get_mask_fn(match_name_fn, params):
    """根据参数路径生成 Optax mask。"""

    mask = jax.tree_util.tree_map_with_path(lambda path, _: match_name_fn(tree_path_to_string(path, sep="/")), params)
    return mask
