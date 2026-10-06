"""FAISS 向量索引封装。

相比旧的 service/faiss.py 有三点不同：
1. 改名。旧文件名 faiss.py 与第三方包 faiss 同名（包内 `import faiss` 读起来像读自己），
   改名 vector_index.py，类名 VectorIndex。
2. 不再在 import 时读盘。旧代码在模块级执行 faiss.read_index("rag_index.index")，
   文件不存在时连 import 都失败；现在由 VectorIndex.load() 显式加载。
3. 索引作为对象持有。旧代码没有索引变量，检索用的是模块级那份，入库时又新建一份，
   两者互不相干。现在 add/search 操作的是同一个 index。
"""
from pathlib import Path

import faiss
import numpy as np

from .config import Settings


class VectorIndex:
    """IndexFlatIP（内积）索引。向量入库前统一做 L2 归一化，此时内积等价于余弦相似度。"""

    def __init__(self, dim: int, index=None):
        self.dim = int(dim)
        self.index = index if index is not None else faiss.IndexFlatIP(self.dim)

    def __len__(self) -> int:
        return self.index.ntotal

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @classmethod
    def load(cls, path, dim: int = None) -> "VectorIndex":
        """从磁盘加载索引。

        文件不存在时返回空索引而不是抛异常：首次启动、还没建过索引的情况不该直接崩。
        """
        path = Path(path)
        if not path.exists():
            fallback = dim if dim is not None else Settings.from_env().embedding_dim
            return cls(fallback)
        index = faiss.read_index(str(path))
        return cls(dim=index.d, index=index)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def add(self, vectors) -> int:
        """向量入库，返回入库后的总量。追加写入，不会清掉已有向量；落盘由调用方 save()。"""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors 需要是二维数组，当前 shape={vectors.shape}")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不符：索引为 {self.dim}，输入为 {vectors.shape[1]}")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        return self.index.ntotal

    def search(self, query_vector, top_k: int = 3):
        """检索最相似的 top_k 条，返回 (scores, ids)，形状均为 (1, k)。索引为空时返回空结果。"""
        if self.index.ntotal == 0 or top_k <= 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)

        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"查询向量维度不符：索引为 {self.dim}，输入为 {query.shape[1]}")
        faiss.normalize_L2(query)
        return self.index.search(query, k=min(int(top_k), self.index.ntotal))
