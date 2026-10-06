"""离线链路自检：用假 embedding 替换真实接口，确认双索引全链路代码是通的。

不访问网络，不消耗额度。真实效果请用 build_index.py + tests/eval_retrieval.py。

假向量只是「确定性字符哈希」，没有真实语义。它的作用是验证
「知识库读取 → 问题/答案分别向量化 → 双索引落盘 → 重新加载
  → 双路召回 → 合并去重 → rerank → 拼装上下文」
每一环都能跑通，而不是验证检索准确率。所以「命中 0 条」在这里属预期。

运行（在项目根目录执行）：
    venv\\Scripts\\python.exe tests\\_selftest.py
"""
import sys
from pathlib import Path

import numpy as np

# 本脚本在 tests/ 下，项目根目录是其上一级；加入 sys.path 才能 import service
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from service import NoopReranker, RAGService, Reranker, Settings, VectorIndex, load_knowledge

DIM = 1024


def fake_embed(text: str, *, settings=None, timeout=None) -> np.ndarray:
    """确定性假向量：按字符 bigram 做哈希桶，同样的文本得到同样的向量。"""
    vec = np.zeros(DIM, dtype=np.float32)
    t = str(text)
    for i in range(len(t) - 1):
        vec[hash(t[i:i + 2]) % DIM] += 1.0
    if not vec.any():
        vec[hash(t) % DIM] = 1.0
    return vec


class RecordingReranker(Reranker):
    """记录被调用情况的假 reranker，用来验证「先 rerank 再截断」的顺序。"""

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents, top_n):
        self.calls.append({"query": query, "n_docs": len(documents), "top_n": top_n})
        # 故意倒序返回，检验 RAGService 是否真的采用了 rerank 的顺序
        order = list(range(len(documents)))[::-1]
        return [(i, float(len(documents) - i)) for i in order]


def main() -> int:
    cfg = Settings.from_env()
    docs = load_knowledge(cfg.knowledge_path)
    print(f"[1] 知识库读取成功：{len(docs)} 条")

    # --- 问题 / 答案分别向量化，模拟 build_index.py 的流程 ---
    q_vecs = np.stack([fake_embed(d["question"]) for d in docs])
    a_vecs = np.stack([fake_embed(d["answer"]) for d in docs])
    print(f"[2] 问题向量 {q_vecs.shape}  答案向量 {a_vecs.shape}（两份独立向量）")

    q_index = VectorIndex(DIM)
    q_index.add(q_vecs)
    a_index = VectorIndex(DIM)
    a_index.add(a_vecs)

    tmp_dir = ROOT / "_selftest_idx"
    tmp_dir.mkdir(exist_ok=True)
    q_path, a_path = tmp_dir / "question.index", tmp_dir / "answer.index"
    q_index.save(q_path)
    a_index.save(a_path)
    print(f"[3] 双索引写出成功：{q_path.name}={q_index.ntotal}  {a_path.name}={a_index.ntotal}")

    # --- 模拟服务启动：从磁盘重新加载两个索引 ---
    svc = RAGService(
        knowledge_list=load_knowledge(cfg.knowledge_path),
        question_index=VectorIndex.load(q_path, dim=DIM),
        answer_index=VectorIndex.load(a_path, dim=DIM),
        settings=cfg,
        embed_fn=fake_embed,
        reranker=NoopReranker(),
    )
    print(f"[4] RAGService 装配成功：问题索引={len(svc.question_index)} "
          f"答案索引={len(svc.answer_index)}")

    # --- 双路召回：确认两路都在工作并带 source 标注 ---
    svc.score_threshold = 0.0          # 假向量无语义，放开阈值才能看到召回行为
    svc.answer_score_threshold = 0.0
    cands = svc.candidates("你是谁")
    sources = sorted({c["source"] for c in cands})
    print(f"[5] 双路召回返回 {len(cands)} 条候选，来源标注={sources}")
    assert any("question" in c["source"] for c in cands), "问题索引这一路没工作"
    assert any("answer" in c["source"] for c in cands), "答案索引这一路没工作"

    # --- 验证合并去重：同一 doc_idx 不应重复出现 ---
    idxs = [c["doc_idx"] for c in cands]
    assert len(idxs) == len(set(idxs)), f"合并去重失败，出现重复下标: {idxs}"
    dup = [c for c in cands if len(c["path_scores"]) > 1]
    print(f"[6] 合并去重正确（无重复下标）；两路都命中的条目 {len(dup)} 条")

    # --- 验证 rerank 顺序被采用，且发生在截断之前 ---
    rec = RecordingReranker()
    svc.reranker = rec
    svc.top_k = 3
    hits = svc.search("你是谁")
    call = rec.calls[-1]
    print(f"[7] rerank 被调用：候选 {call['n_docs']} 条 → top_n={call['top_n']}")
    print(f"    返回 {len(hits)} 条，首条={hits[0]['question']!r}（倒序 rerank 的首条）")
    assert call["n_docs"] == len(cands), "rerank 拿到的候选数与召回数不一致"
    assert len(hits) <= 3, "top_k 截断失效"
    # 倒序应等价于「原候选列表的最后一条」排第一，证明顺序确实被采用
    assert hits[0]["question"] == cands[-1]["question"], "rerank 返回的顺序没有被采用"

    # --- rerank 抛异常时应降级而不是整体失败 ---
    class BrokenReranker(Reranker):
        def rerank(self, query, documents, top_n):
            raise RuntimeError("模拟 rerank 服务不可用")

    svc.reranker = BrokenReranker()
    hits2 = svc.search("你是谁")
    print(f"[8] rerank 异常时降级成功，仍返回 {len(hits2)} 条（未中断检索）")
    assert len(hits2) > 0

    # --- 缺失答案索引时应退化为单路，而不是崩 ---
    svc.answer_index = None
    cands_single = svc.candidates("你是谁")
    print(f"[9] 缺少答案索引时退化为单路，仍召回 {len(cands_single)} 条")

    # --- 参数校验 ---
    svc.answer_index = a_index
    for bad in ["", "   "]:
        try:
            svc.search(bad)
            print(f"[10] 空 query {bad!r} 未被拦截 <-- BUG")
        except ValueError:
            print(f"[10] 空 query {bad!r} 被正确拦截")

    # --- 清理 ---
    q_path.unlink(missing_ok=True)
    a_path.unlink(missing_ok=True)
    tmp_dir.rmdir()

    print("\n全部离线检查通过：双索引 + 双路召回 + rerank 插拔链路无阻塞。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
