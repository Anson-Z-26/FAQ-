"""索引清单（manifest）：把「FAISS 行号」和「知识条目」的对应关系显式记录下来。

## 为什么需要它

向量检索返回的是**行号**，而业务需要的是「哪条知识」。旧代码靠一条隐含约定：

    FAISS 第 i 行  ==  knowledge_list[i]

这在「一条知识一个向量」时刚好成立，但只要出现下面任一情况就断裂：

  1. 一条知识有多个问法变体（aliases）→ 问题索引行数 > 知识条数
  2. 数据库返回顺序不确定（没有 ORDER BY）
  3. 增删了一条知识，后续行号整体偏移

断裂的后果**不是报错，是静默返回错误答案** —— 分数、日志、排序全都看起来正常，
只有答案内容是错的。这类 bug 极难定位，所以必须用显式映射替代隐含约定。

## 清单里存什么

  - 每行属于哪个知识条目（question_rows / answer_rows）
  - embedding 模型名与维度（换模型后旧索引作废，必须能检测到）
  - 每条知识的内容指纹（hash，用于检测「索引建完后知识库又改了」）
  - 构建时间与格式版本

## 与「增量重建」的关系

内容指纹让「只重建改动的条目」成为可能，但本版本只做**校验**不做增量：
发现 hash 变化就明确报告「索引已过期，请重建」，而不是悄悄用旧向量。
增量重建留作后续优化。
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

FORMAT_VERSION = 2

# 参与指纹计算的字段。顺序固定，保证同一内容总是得到同一 hash。
_HASH_FIELDS = ("question", "aliases", "answer")


def entry_fingerprint(doc: dict) -> str:
    """计算一条知识的内容指纹。

    只取影响检索结果的字段：question / aliases / answer。
    改动 source、url、updated 不影响向量，不该触发重建。
    """
    payload = {k: doc.get(k) for k in _HASH_FIELDS}
    # aliases 是数组，统一成有序字符串，避免键序不同导致 hash 不同
    if isinstance(payload.get("aliases"), list):
        payload["aliases"] = [str(a) for a in payload["aliases"]]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class IndexManifest:
    """索引与知识条目的对应关系快照。"""

    embedding_model: str
    embedding_dim: int
    knowledge_count: int
    # 每个 FAISS 行对应的知识 id，长度必须等于对应索引的 ntotal
    question_rows: list = field(default_factory=list)
    answer_rows: list = field(default_factory=list)
    # [{"id": ..., "hash": ...}, ...]，按知识库顺序
    entries: list = field(default_factory=list)
    built_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    format_version: int = FORMAT_VERSION

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "knowledge_count": self.knowledge_count,
            "built_at": self.built_at,
            "question_rows": self.question_rows,
            "answer_rows": self.answer_rows,
            "entries": self.entries,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "IndexManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"索引清单不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            embedding_model=data.get("embedding_model", ""),
            embedding_dim=int(data.get("embedding_dim", 0)),
            knowledge_count=int(data.get("knowledge_count", 0)),
            question_rows=list(data.get("question_rows") or []),
            answer_rows=list(data.get("answer_rows") or []),
            entries=list(data.get("entries") or []),
            built_at=data.get("built_at", ""),
            format_version=int(data.get("format_version", 0)),
        )

    # ------------------------------------------------------------------
    def validate(self, *, embedding_model: str, embedding_dim: int,
                 knowledge_list: list, question_ntotal: int = None,
                 answer_ntotal: int = None) -> list:
        """校验清单与当前配置/知识库是否一致，返回问题列表（空表示通过）。

        刻意把所有问题都收集出来一次性报告，而不是遇到第一个就抛异常 ——
        重建索引时你希望一次看到全部原因，而不是修一个再跑一次。
        """
        problems = []

        if self.format_version != FORMAT_VERSION:
            problems.append(
                f"清单格式版本 {self.format_version} 与当前代码 ({FORMAT_VERSION}) 不符，"
                f"请重建索引"
            )

        if self.embedding_model != embedding_model:
            problems.append(
                f"索引用的 embedding 模型是 {self.embedding_model!r}，"
                f"当前配置是 {embedding_model!r}；换模型后旧向量不可复用，请重建索引"
            )

        if self.embedding_dim != embedding_dim:
            problems.append(
                f"索引维度 {self.embedding_dim} 与配置 {embedding_dim} 不一致"
            )

        if self.knowledge_count != len(knowledge_list):
            problems.append(
                f"建索引时知识库有 {self.knowledge_count} 条，现在有 "
                f"{len(knowledge_list)} 条；请重建索引"
            )

        # 行数与索引实际条数对不上 → 说明索引文件被替换或损坏
        if question_ntotal is not None and len(self.question_rows) != question_ntotal:
            problems.append(
                f"问题索引实际有 {question_ntotal} 行，清单记录了 "
                f"{len(self.question_rows)} 行，两者不一致"
            )
        if answer_ntotal is not None and len(self.answer_rows) != answer_ntotal:
            problems.append(
                f"答案索引实际有 {answer_ntotal} 行，清单记录了 "
                f"{len(self.answer_rows)} 行，两者不一致"
            )

        # 清单里引用的 id 必须都还在知识库里
        known_ids = {d.get("id") for d in knowledge_list if d.get("id")}
        referenced = set(self.question_rows) | set(self.answer_rows)
        missing = referenced - known_ids
        if missing:
            problems.append(
                f"清单引用了知识库中已不存在的 id: {sorted(missing)[:8]}"
                f"{' 等' if len(missing) > 8 else ''}"
            )

        # 逐条比对内容指纹，检测「索引建完后知识内容又被改了」
        manifest_hashes = {e.get("id"): e.get("hash") for e in self.entries}
        stale = []
        for doc in knowledge_list:
            did = doc.get("id")
            if not did:
                continue
            old = manifest_hashes.get(did)
            if old and old != entry_fingerprint(doc):
                stale.append(did)
        if stale:
            problems.append(
                f"以下 {len(stale)} 条知识在索引构建后被修改过，向量已过期: "
                f"{sorted(stale)[:8]}{' 等' if len(stale) > 8 else ''}；请重建索引"
            )

        return problems

    # ------------------------------------------------------------------
    def id_for_question_row(self, row: int) -> str | None:
        if 0 <= row < len(self.question_rows):
            return self.question_rows[row]
        return None

    def id_for_answer_row(self, row: int) -> str | None:
        if 0 <= row < len(self.answer_rows):
            return self.answer_rows[row]
        return None


def build_manifest(*, embedding_model: str, embedding_dim: int,
                   knowledge_list: list, question_rows: list,
                   answer_rows: list) -> IndexManifest:
    """由建索引时的实际数据生成清单。"""
    return IndexManifest(
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        knowledge_count=len(knowledge_list),
        question_rows=list(question_rows),
        answer_rows=list(answer_rows),
        entries=[
            {"id": d.get("id"), "hash": entry_fingerprint(d)}
            for d in knowledge_list
        ],
    )
