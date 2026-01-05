"""
部件树预处理：
- 将每个节点的 `name` 统一为 List[str]
- 将“单子节点链”（每层只有 1 个 child 的连续结构）压缩为单个节点：
  - 压缩后节点的 `name` 为链上所有节点名字的累积 List[str]（保序去空）
  - `maskId` 优先继承更深层节点的（若存在）
  - 子节点继承链末端节点的 children

注意：该预处理应在所有后续操作（region 计算 / caption / QA）之前执行。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _collect_children(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    ch = node.get("children")
    if isinstance(ch, list):
        return [x for x in ch if isinstance(x, dict)]
    return []


def _norm_name_to_list(node: Dict[str, Any], *, object_id: Optional[str] = None) -> None:
    """
    将 node["name"] 规范化为 List[str]（保序，去空白）。
    """
    name = node.get("name")
    out: List[str] = []
    if isinstance(name, str):
        s = name.strip()
        if s:
            out = [s]
    elif isinstance(name, list):
        for x in name:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)

    # 极少数情况下根节点 name 为空：用 object_id 兜底
    if not out and object_id:
        out = [str(object_id)]

    node["name"] = out


def _merge_names_preserve_order(a: List[str], b: List[str]) -> List[str]:
    """
    合并两个 name list（保序去重）。
    """
    seen = set()
    out: List[str] = []
    for x in (a or []) + (b or []):
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _append_name_candidates(node: Dict[str, Any], new_cands: List[str]) -> None:
    """
    追加到 node["name_candidates"]（保序去重）。
    """
    if not new_cands:
        return
    base = []
    if isinstance(node.get("name_candidates"), list):
        base = [x.strip() for x in node.get("name_candidates") if isinstance(x, str) and x.strip()]
    merged = _merge_names_preserve_order(base, new_cands)
    if merged:
        node["name_candidates"] = merged


def _compress_single_child_chain(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归压缩单子节点链：返回压缩后的 node（原地修改并返回自身引用）。
    """
    # 先递归处理 children
    children = _collect_children(node)
    if children:
        node["children"] = [_compress_single_child_chain(c) for c in children]

    # 再在当前节点做链压缩：只要当前节点有且仅有一个子节点，就把子节点“合并进来”
    while True:
        children = _collect_children(node)
        if len(children) != 1:
            break

        child = children[0]
        # 合并 name：链上名字累积到当前节点
        node_name = node.get("name") if isinstance(node.get("name"), list) else []
        child_name = child.get("name") if isinstance(child.get("name"), list) else []
        # 重要：把更深层（更具体）的名字放前面，保持后续逻辑用 candidates[0] 时依然偏向具体部件名
        node["name"] = _merge_names_preserve_order(child_name, node_name)

        # 合并候选名：同样让 child 的候选优先
        if isinstance(child.get("name_candidates"), list):
            parent_cands = []
            if isinstance(node.get("name_candidates"), list):
                parent_cands = [x for x in node.get("name_candidates") if isinstance(x, str)]
            child_cands = [x for x in child.get("name_candidates") if isinstance(x, str)]
            merged_cands = _merge_names_preserve_order(child_cands, parent_cands)
            if merged_cands:
                node["name_candidates"] = merged_cands

        # 记录被压缩的 nodeId（可选，便于 debug / 追溯）
        merged_ids: List[str] = []
        if isinstance(node.get("_merged_node_ids"), list):
            merged_ids.extend([str(x) for x in node.get("_merged_node_ids") if x is not None])
        else:
            if node.get("id") is not None:
                merged_ids.append(str(node.get("id")))
        if isinstance(child.get("_merged_node_ids"), list):
            merged_ids.extend([str(x) for x in child.get("_merged_node_ids") if x is not None])
        else:
            if child.get("id") is not None:
                merged_ids.append(str(child.get("id")))
        # 去重保序
        merged_ids = _merge_names_preserve_order(merged_ids, [])
        if merged_ids:
            node["_merged_node_ids"] = merged_ids

        # maskId：优先用更深层 child 的（更具体）
        if child.get("maskId", None) is not None:
            node["maskId"] = child.get("maskId")

        # children：继承链末端的 children（child 已经递归压缩过）
        node["children"] = _collect_children(child)

        # 保留 node 自身 id（更稳定）；不把 child.id 覆盖上来
        # 继续 while，直到不再是单子节点

    return node


def preprocess_part_tree(root: Dict[str, Any], *, object_id: Optional[str] = None) -> Dict[str, Any]:
    """
    对部件树做统一预处理（原地修改并返回 root）。
    """
    if not isinstance(root, dict):
        raise TypeError(f"root 必须是 dict, got {type(root)}")

    # 先全树把 name 变成 list
    def rec_norm(n: Dict[str, Any], is_root: bool) -> None:
        _norm_name_to_list(n, object_id=object_id if is_root else None)
        for c in _collect_children(n):
            rec_norm(c, False)

    rec_norm(root, True)

    # 再压缩链
    _compress_single_child_chain(root)
    return root


