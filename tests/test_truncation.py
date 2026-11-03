"""验证 rerank 后的「动态截断」：按相对分数裁掉弱相关条目。

## 要解决的问题

rerank 分数是**本次请求内的相对分数**（官方明确说明不可跨请求比较）。
旧实现只用固定绝对门槛 0.40，导致：

    第1名 0.95 的请求：0.45 的「沾点边」条目也能进来，白占上下文
    第1名 0.50 的请求：同样的 0.40 门槛反而相对严苛

改为「绝对门槛 + 相对门槛（>= 最佳分 × ratio）」双闸门：
  - 绝对门槛：这条够不够格被回答
  - 相对门槛：这条和最佳答案差多少，差太多就不打扰大模型

## 本脚本用可控的假 reranker 测边界

真实 rerank 分数不可预测，所以这里注入固定分数，逐一验证：
  1. 高置信查询（第1名 0.95）→ 只保留接近 0.95 的
  2. 低置信查询（第1名 0.45）→ 绝对门槛兜底，不因 ratio 过严而裁空
  3. 保底条数（min_keep）生效，不会一条不留
  4. ratio=0 时退化为旧行为（只按绝对门槛 + top_k）
  5. 所有分数都低于绝对门槛 → 返回 0 条（正确拒答）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import NoopReranker, RAGService, Reranker, Settings, VectorIndex

DIM = 8

DOCS = [
    {"id": f"D{i}", "question": f"问题{i}", "answer": f"答案{i}", "aliases": []}
    for i in range(6)
]


class FixedReranker(Reranker):
    """按给定分数表返回固定排序，用于测截断边界。"""

    def __init__(self, scores):
        self.scores = scores

    def rerank(self, query, documents, top_n):
        order = sorted(range(len(self.scores)), key=lambda i: -self.scores[i])
        return [(i, float(self.scores[i])) for i in order]


def fake_embed(text, *, settings=None, **kw):
    v = np.zeros(DIM, dtype=np.float32)
    for ch in str(text):
        v[ord(ch) % DIM] += 1.0
    if not v.any():
        v[0] = 1.0
    return v


def build(cfg, *, ratio, min_keep, threshold=0.40):
    idx = VectorIndex(DIM)
    idx.add(np.stack([fake_embed(d["question"]) for d in DOCS]))
    return RAGService(
        knowledge_list=DOCS,
        question_index=idx,
        answer_index=None,
        settings=cfg,
        embed_fn=fake_embed,
        reranker=NoopReranker(),
        top_k=5,
        score_threshold=0.0,     # 放开向量阈值，让全部条目进候选
        recall_k=6,
        rerank_score_threshold=threshold,
        rerank_relative_ratio=ratio,
        rerank_min_keep=min_keep,
        question_row_ids=[d["id"] for d in DOCS],
    )


def run_case(name, scores, ratio, min_keep, threshold, expect_n, expect_ids=None):
    cfg = Settings.from_env()
    svc = build(cfg, ratio=ratio, min_keep=min_keep, threshold=threshold)
    # 直接替换 reranker（search 内部用的是 self.reranker）
    svc.reranker = FixedReranker(scores)
    # 让 search 走 reranker 分支而非 Noop：NoopReranker 判断基于实例类型
    hits = svc.search("问题0")
    ids = [h["id"] for h in hits]
    got = [round(h["rerank_score"], 3) for h in hits]
    ok = len(hits) == expect_n and (expect_ids is None or ids == expect_ids)
    print(f"  {'✓' if ok else '✗'} {name}")
    print(f"     分数 {scores}  ratio={ratio} min_keep={min_keep} 门槛={threshold}")
    print(f"     结果 {len(hits)} 条 {ids}  分数 {got}")
    if not ok:
        print(f"     期望 {expect_n} 条 {expect_ids or ''}")
    return ok


def main() -> int:
    cfg = Settings.from_env()
    print("注：RAGService.search 用 isinstance(reranker, NoopReranker) 判断是否启用门槛，\n"
          "    这里注入的 FixedReranker 不是 Noop，所以走门槛分支。\n")

    all_ok = True

    print("[1] 高置信查询：第1名 0.95，ratio=0.6 -> 相对门槛 0.57")
    #  0.95 ✓  0.78 ✓  0.55 ✗  0.51 ✗  0.44 ✗  0.41 ✗
    all_ok &= run_case("只保留 0.95 与 0.78", [0.95, 0.78, 0.55, 0.51, 0.44, 0.41],
                       ratio=0.6, min_keep=1, threshold=0.40, expect_n=2)

    print("\n[2] 低置信查询：第1名 0.45，ratio=0.6 -> 相对门槛 0.27，但绝对门槛 0.40 更严")
    #  0.45 ✓  0.42 ✓  0.36 ✗(绝对)  0.30 ✗  0.25 ✗  0.20 ✗
    all_ok &= run_case("绝对门槛兜底，保留 2 条", [0.45, 0.42, 0.36, 0.30, 0.25, 0.20],
                       ratio=0.6, min_keep=1, threshold=0.40, expect_n=2)

    print("\n[3] 保底：第1名 0.92，第2名 0.20（远低于相对门槛 0.552）")
    #  0.92 ✓  0.20 ✗(绝对门槛也不到) -> 只有 1 条
    all_ok &= run_case("绝对门槛已挡掉第2名", [0.92, 0.20, 0.18, 0.15, 0.12, 0.10],
                       ratio=0.6, min_keep=1, threshold=0.40, expect_n=1)

    print("\n[4] 保底=2：第1名 0.92，第2名 0.55（0.55 > 0.552? 否，应被裁）")
    # 相对门槛 = 0.92*0.6 = 0.552；0.55 < 0.552 -> 裁掉，但 min_keep=2 保住
    all_ok &= run_case("min_keep=2 保住第2条", [0.92, 0.55, 0.53, 0.50, 0.48, 0.45],
                       ratio=0.6, min_keep=2, threshold=0.40, expect_n=2)

    print("\n[5] ratio=0 关闭动态截断 -> 退化为旧行为（过绝对门槛就保留到 top_k）")
    #  0.95 0.78 0.55 0.51 0.44 都过 0.40 -> top_k=5 取满
    all_ok &= run_case("保留 5 条（top_k 上限）", [0.95, 0.78, 0.55, 0.51, 0.44, 0.41],
                       ratio=0.0, min_keep=0, threshold=0.40, expect_n=5)

    print("\n[6] 全部低于绝对门槛 -> 0 条（正确拒答）")
    all_ok &= run_case("拒答", [0.30, 0.28, 0.25, 0.22, 0.20, 0.18],
                       ratio=0.6, min_keep=1, threshold=0.40, expect_n=0)

    print("\n[7] top_k 上限仍然生效：ratio=0 且分数都很高")
    all_ok &= run_case("最多 5 条", [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
                       ratio=0.0, min_keep=0, threshold=0.40, expect_n=5)

    print("\n" + ("全部检查通过：动态截断边界正确。" if all_ok else "存在失败项 ✗"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
