"""逐条复核确认：按稳定条款 ID 对存活条款做「保存时状态」确认。

确认绑定四要素（一律以服务器**已保存**的草稿快照为准）：

- **正文**：条款原文（不含编号标记）；
- **父级**：父级稳定身份——父路径上有真实条款时为其稳定 ID，否则为虚父
  路径标记（如 ``virtual:3``），顶层为 ``""``；
- **引用目标**：逐处引用的状态与目标稳定 ID 集合（不含重排后的新编号与
  改写文本）；
- **术语绑定**：本条款内定义（语义键 + 状态）与每处使用绑定的定义语义键
  （语义键 = 术语名 + 作用域 + 所在块稳定 ID + 同块同名序，不随正文漂移）。

四要素任一项变化（正文 / 父级修改，或重排、删除、恢复、应用版本差异导致
绑定改变）→ 仅相关条款的确认过期；**单纯编号变化**（重排或其他条款删除
带来的新编号、引用改写文本变化）不进入指纹，不撤销确认。

确认记录存于 ``review_confirmations`` 表（按 ``(draft_id, clause_id)`` 唯一），
有效性通过「保存的指纹 vs 当前指纹」惰性比对得出，无需在保存时主动失效。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .parser import parse_contract
from .references import analyze, cascade_deleted
from .terms import review_terms


def _parent_identity(path: tuple[int, ...],
                     by_path: dict[tuple[int, ...], str]) -> str:
    """父级稳定身份：父路径存在真实条款 → 其稳定 ID；虚父 → 路径标记。"""
    if len(path) <= 1:
        return ""
    ppath = path[:-1]
    pid = by_path.get(ppath)
    if pid is not None:
        return f"clause:{pid}"
    return "virtual:" + "-".join(str(p) for p in ppath)


def clause_fingerprints(content: str, order_in: list[str],
                        deleted_in: list[str]):
    """对当前快照逐条计算存活条款的确认指纹。

    返回 ``(survivors, fingerprints, labels)``：

    - ``survivors``：重排后存活条款 id（新顺序先序）；
    - ``fingerprints``：``clause_id -> 指纹哈希``；
    - ``labels``：``clause_id -> 展示信息``（原编号 / 新编号 / 标题 / 位置）。
    """
    preamble, clauses = parse_contract(content)
    known = {c.id for c in clauses}
    order = [cid for cid in order_in if cid in known]
    deleted = [cid for cid in deleted_in if cid in known]

    ref_result = analyze(clauses, order, deleted)
    deleted_set = cascade_deleted(clauses, deleted)
    term_result = review_terms(preamble, clauses, deleted_set)

    by_id = {c.id: c for c in clauses}
    by_path: dict[tuple[int, ...], str] = {}
    for c in clauses:
        by_path.setdefault(c.path, c.id)

    # 术语定义语义键：(名称, 作用域, 所在块稳定 ID, 同块同名序) —— 不随正文
    # 漂移（与版本差异审计的对齐口径一致）
    doc_index = {c.id: i for i, c in enumerate(clauses)}
    def_keys: dict[str, list] = {}
    counters: dict[tuple, int] = {}
    definitions = sorted(
        term_result["definitions"],
        key=lambda d: (doc_index.get(d["block_id"], -1), d["start"]),
    )
    for d in definitions:
        base = (d["name"], d["scope"], d["block_id"])
        ordinal = counters.get(base, 0)
        counters[base] = ordinal + 1
        def_keys[d["id"]] = [d["name"], d["scope"], d["block_id"], ordinal]

    blocks = {b["id"]: b for b in term_result["blocks"]}
    ref_view = {c["id"]: c for c in ref_result["clauses"]}

    survivors = list(ref_result["order"])
    fingerprints: dict[str, str] = {}
    labels: dict[str, dict] = {}
    for cid in survivors:
        c = by_id[cid]
        view = ref_view[cid]
        refs = [
            {
                "text": m["text"],
                "status": m["status"],
                "targets": sorted(m.get("target_ids") or []),
            }
            for m in view["refs"]
        ]
        defs: list[dict] = []
        uses: list[dict] = []
        for m in blocks.get(cid, {}).get("marks", []):
            if m["kind"] == "def":
                defs.append({
                    "key": def_keys.get(m["occ_id"]),
                    "status": m["status"],
                })
            else:
                uses.append({
                    "name": m["name"],
                    "quoted": bool(m.get("quoted")),
                    "status": m["status"],
                    "bound": sorted(
                        def_keys[t] for t in m.get("target_ids", [])
                        if t in def_keys
                    ),
                })
        payload = {
            "body": c.text,
            "parent": _parent_identity(c.path, by_path),
            "refs": refs,
            "defs": defs,
            "uses": uses,
        }
        fingerprints[cid] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        labels[cid] = {
            "label": c.label,
            "new_label": view["new_label"],
            "heading": c.heading,
            "new_position": view["new_position"],
        }
    return survivors, fingerprints, labels


def _conf_map(rows: Iterable[Any]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        d = dict(r)
        out[d["clause_id"]] = d
    return out


def build_review_status(content: str, order: list[str], deleted: list[str],
                        confirmations: Iterable[Any]) -> dict:
    """逐条复核状态：对比当前指纹与已保存确认，给出 pending / valid / stale。"""
    survivors, fps, labels = clause_fingerprints(content, order, deleted)
    conf = _conf_map(confirmations)
    items: list[dict] = []
    counts = {"total": 0, "pending": 0, "valid": 0, "stale": 0}
    for cid in survivors:
        row = conf.get(cid)
        if row is None:
            status = "pending"
        elif row["fingerprint"] == fps[cid]:
            status = "valid"
        else:
            status = "stale"
        counts["total"] += 1
        counts[status] += 1
        items.append({
            "clause_id": cid,
            "status": status,
            "confirmed_at": row["confirmed_at"] if row else None,
            "confirmed_revision": row["revision"] if row else None,
            **labels[cid],
        })
    return {"items": items, "counts": counts}


def review_issues(survivors: list[str], fingerprints: dict[str, str],
                  labels: dict[str, dict],
                  confirmations: Iterable[Any]) -> list[dict]:
    """签发前复查：缺失 / 已过期的确认清单（供 422 明细与定位）。"""
    conf = _conf_map(confirmations)
    out: list[dict] = []
    for cid in survivors:
        row = conf.get(cid)
        if row is None:
            status = "pending"
        elif row["fingerprint"] != fingerprints[cid]:
            status = "stale"
        else:
            continue
        lab = labels[cid]
        out.append({
            "clause_id": cid,
            "status": status,
            "label": lab["label"],
            "new_label": lab["new_label"],
            "heading": lab["heading"],
            "message": (
                "尚未提交复核确认" if status == "pending"
                else "确认已过期：正文、父级、引用目标或术语绑定已变化，"
                     "需重新复核确认"
            ),
        })
    return out
