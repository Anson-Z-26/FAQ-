# 智能问答助手

基于知识库的 RAG 语义检索问答服务，带配套前端界面。

## 它做什么

用户在页面上提问 → 系统从知识库中检索最相关的条目 → 交给大模型生成回答。

生成的回答**严格依据检索到的材料**，材料不足时明确说「无法回答」而不是编造。
界面上会同时展示回答和它的依据（可展开查看命中了哪些知识条目），
方便核对是检索错了还是模型答偏了。

## 架构

```
提问 → 向量化
        ├─ 查「问题索引」→ 候选 A（问法匹配）
        └─ 查「答案索引」→ 候选 B（内容匹配）
      合并去重 → rerank 精排 → 取 top_k → 大模型生成 → 回答 + 依据
```

整条链路是**串行**的，各阶段耗时（实测，13 条知识规模）：

| 阶段 | 耗时 |
|---|---|
| embedding | 约 0.7s |
| 检索 + rerank | 约 1.8s |
| 大模型生成 | 约 1.4s |
| **合计** | **约 3~4s** |

命中为空时会跳过生成（不调模型），响应降到 1s 以内。
多轮追问会多一次「查询改写」调用，约多 0.9s。

## 多轮对话

前端会自动带上 `session_id`，追问可以省略主语：

```
用户：宿舍条件怎么样
用户：它有空调吗          ← 系统理解为「宿舍有空调吗」后检索
```

**关键在查询改写。** 检索层只看到「它有空调吗」，离开上文和任何知识都匹配不上
（实测命中 0 条）。所以检索前先用模型结合历史把追问补全成独立问题：

| 追问 | 改写为 | 命中 |
|---|---|---|
| 它有空调吗 | 宿舍有空调吗 | 0 → 2 |
| 它在哪里 | 西安电子科技大学在哪里 | 0 → 5 |
| 那食堂呢 | 食堂怎么样 | 0 → 1 |
| 宿舍条件怎么样（已完整） | 原样返回 | 不变 |

改写有两条硬约束：**已经是完整问题时原样返回**（避免把「学校在哪」改坏），
**只补全指代、不引入上文没有的信息**。界面上会显示「已结合上文理解为：…」，
改写错了用户能立刻发现。

会话使用 `keep_history` 控制：

```bash
# 单轮（默认）：不读也不写历史
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" -d '{"query":"学校在哪"}'

# 多轮：首次不传 session_id，服务端新建并在响应里返回
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"宿舍条件怎么样","keep_history":true}'

# 追问：带上拿到的 session_id
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"它有空调吗","keep_history":true,"session_id":"<上一步返回的>"}'
```

> **会话当前存在内存里**，有两个边界：`uvicorn --workers 2` 时多个进程各存一份，
> 同一会话落到不同 worker 会丢历史；进程重启历史即丢。
> 单进程开发/小规模部署够用，要多 worker 需换 Redis。

**为什么问题和答案分开建索引**

早期做法是把 `question + answer` 拼成一条文本去向量化，有两个问题：

1. **语义稀释**：答案文字会冲淡问题向量的语义。本知识库条目大多以
   「西安电子科技大学」开头，区分度本来就低，再混入答案就更挤在一起。
2. **答案向量被浪费**：用户问法和知识库问法经常对不上（用户问「风景美不美」，
   知识库里写的是「校园环境怎么样」），但答案正文里可能有相关信息。
   把答案单独建索引，这条就能被捞回来。

**为什么需要 rerank**

向量检索是「双塔」结构：问题和文档各自独立编码，编码时互相看不见，
会丢掉细粒度交互信息。rerank 是 cross-encoder，把 query 和候选文档
拼在一起打分，排序远比向量点积准。它的分数还能用来判断「该不该回答」。

**为什么关掉大模型的深度思考**

`deepseek-v4.1-flash` 默认开启思考模式，会先输出一大段思维链。实测：

```
默认思考                约 2.2s，产生 63 个 reasoning token
enable_thinking=false   约 1.6s，0 个 reasoning token
```

RAG 的材料和问题都摆在明面上，不需要多步推理；思维链还按输出 token 计费，
所以默认关闭（`LLM_DISABLE_THINKING=true`）。

## 目录结构

```
├── main.py                 启动入口（uvicorn main:app）
├── build_index.py          建索引：问题、答案各建一个
├── requirements.txt        运行依赖
├── .env.example            配置模板（复制成 .env 后填值）
│
├── api/
│   └── rag_service.py      HTTP 层：/get_answer /ask /health /ui-config + 静态托管
├── service/                业务层（不依赖 web 框架，可单独测试）
│   ├── config.py           配置读取与校验（Settings 为不可变快照）
│   ├── embedding.py        embedding 客户端（自动分批，接口单次上限 10 条）
│   ├── vector_index.py     FAISS 索引封装（内积 + L2 归一化 = 余弦相似度）
│   ├── rerank.py           rerank 抽象 + Noop 实现 + qwen3-rerank 实现
│   ├── llm.py              大模型生成与查询改写（提示词约束 + Noop 实现）
│   ├── session.py          多轮会话存储（内存 + TTL + 轮数上限）
│   └── rag.py              双路召回、合并去重、精排、改写、生成
├── frontend/               零构建纯静态前端（原生 HTML/CSS/JS）
├── db/
│   └── data.json           知识库正文（人工维护）
├── index/                  向量索引（由 build_index.py 生成，不入库）
└── tests/                  验证与评测脚本
```

## 快速开始

```bash
# 1. 安装依赖
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 配置
copy .env.example .env      # 然后填入 DASHSCOPE_API_KEY

# 3. 验证 Key 是否可用
venv\Scripts\python.exe tests\check_api.py

# 4. 建索引（改了知识库后都要重跑）
venv\Scripts\python.exe build_index.py

# 5. 启动
venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

打开 <http://127.0.0.1:8000> 即为界面。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 重定向到界面 `/ui/` |
| GET | `/health` | 服务与索引状态（**索引数不等于知识数时 `ready=false`**） |
| GET | `/ui-config` | 前端启动配置（标题、查询地图地址） |
| POST | `/ask` | **问答主入口**：检索 + 生成回答 + 依据 |
| POST | `/get_answer` | 只做检索，返回拼好的上下文字符串（不调模型） |
| GET | `/docs` | 自动生成的 API 文档 |

`/ask` 的 `generate` 参数可覆盖默认行为：

```bash
# 默认：跟随 LLM_ENABLED（开启则生成回答）
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"学校在哪"}'

# 只检索、不调模型（调试检索效果时用，不产生生成费用）
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"学校在哪","generate":false}'
```

返回结构：

```json
{
  "answer": "学校位于陕西省西安市……",   // 生成的回答
  "generated": true,                    // 是否由大模型生成
  "generation_error": "",               // 生成失败时的原因（此时 answer 退回材料原文）
  "matched": true,                      // 是否检索到内容
  "hit_count": 3,
  "hits": [ { "question": "...", "answer": "...", "score": 0.61, "rerank_score": 0.61 } ]
}
```

## 测试与评测脚本

均在项目根目录执行：

| 脚本 | 作用 | 是否消耗 API 额度 |
|---|---|---|
| `tests/check_api.py` | 验证 API Key 与接口连通性 | 极少（一条请求） |
| `tests/_selftest.py` | 离线全链路自检（假 embedding） | 不消耗 |
| `tests/test_dedupe.py` | 验证按答案去重逻辑 | 不消耗 |
| `tests/check_ui_contract.py` | 检查 HTML/JS/CSS 引用是否对得上 | 不消耗 |
| `tests/eval_retrieval.py` | 检索效果评测（Top-1、命中率、误报） | 消耗 |
| `tests/check_llm.py` | 验证大模型生成链路与拒答行为（需先启动服务） | 消耗 |
| `tests/check_multiturn.py` | 验证多轮指代消解、会话隔离、重置（需先启动服务） | 消耗 |
| `tests/check_ui.py` | 前端与接口联调（需先启动服务） | 消耗 |

## 调参须知（重要）

**阈值是按实测分数分布定的，不是通用值。** 换 embedding 模型或大改知识库后，
分数的绝对尺度会变（实测 `text-embedding-v3` 与 `v4` 的尺度就不一样，
互相照搬会全错），必须重新扫：

```
分别测「有效查询的最低分」和「无关查询的最高分」，阈值取两者之间。
若两者重叠，说明该路无法用单一阈值干净分离，只能取舍。
```

`.env` 里有三个独立阈值，它们**不同源，不能混用**：

- `SCORE_THRESHOLD` —— 问题索引的向量相似度
- `ANSWER_SCORE_THRESHOLD` —— 答案索引的向量相似度
- `RERANK_SCORE_THRESHOLD` —— rerank 分数（仅在开启 rerank 时生效，
  且它才是「该不该回答」的实际门槛）

跑 `tests/eval_retrieval.py` 可以看到当前配置下的命中率与误报率。

## 知识库格式

`db/data.json` 为数组，每条需带 `question` 和 `answer`：

```json
[
  { "question": "学校在哪", "answer": "西安电子科技大学位于陕西省西安市……" }
]
```

**补充问法变体时务必拆成多行**，不要塞进同一个 `question`：

```json
{ "question": "宿舍条件怎么样", "answer": "宿舍条件良好……" },
{ "question": "宿舍有空调吗",   "answer": "宿舍条件良好……" },
{ "question": "住宿环境好不好", "answer": "宿舍条件良好……" }
```

原因：向量化会把整段文本压成一个点。多行时每个变体是一个**尖锐**向量，
各自精确对应一种问法；合并成一行则语义被平均摊薄，反而比不加更差
（实测 Top-1：多行 5/5 vs 单行 0/5）。同答案的多行会在检索时自动去重，
不会重复占用 `top_k`。

## 后续计划

- [ ] 回答流式输出（当前是拿到完整回答后一次性返回，生成阶段约 1.4s 期间无反馈）
- [ ] 会话持久化（当前内存存储，重启即丢，多 worker 不通用；可换 Redis）
- [ ] 知识库接入数据库存储（目前是 `data.json` 静态文件）

## 已知限制

- **端到端延迟约 3~4 秒**：embedding、rerank、生成是串行三次网络请求。
  多轮追问再加一次改写调用（约 +0.9s）。
  其中「检索 + rerank」约 1.8s 是最大头（候选全部过 rerank 才准）。
  想压延迟可以调小 `RECALL_K`，但要重新跑评测确认召回没掉。
- **多轮会话存在内存里**：`uvicorn --workers N`（N>1）时同一会话落到不同
  worker 会丢历史；重启即丢。
- 当前知识库规模小且问法同质（多条以同一校名开头），
  向量分数挤在窄带里，部分无关提问与有效提问的分数会重叠，
  需要靠 rerank 兜底。
- 大模型可能仍会轻微改写措辞（如把「西安电子科技大学」简称成「学校」），
  提示词已要求只用材料内容，但不能保证 100% 逐字忠实。
- 查询改写依赖模型判断，偶有把完整问题误改的可能；界面上会显示改写了什么，
  便于用户发现。
- 前端未做鉴权，`/ask` 等接口对外暴露，**不要直接部署到公网**。
