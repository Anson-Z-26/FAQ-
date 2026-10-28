/* ==========================================================================
   智能问答助手 · 前端逻辑
   零依赖原生 JS。后端接口：
     GET  /ui-config  读取标题、查询地图地址等配置
     GET  /health     服务与索引状态
     POST /ask        结构化检索结果
   ========================================================================== */
'use strict';

const $ = (id) => document.getElementById(id);

const el = {
  title: $('app-title'),
  mapBtn: $('map-btn'),
  statusBtn: $('status-btn'),
  statusDot: $('status-dot'),
  statusText: $('status-text'),
  banner: $('banner'),
  messages: $('messages'),
  emptyState: $('empty-state'),
  suggestions: $('suggestions'),
  form: $('ask-form'),
  input: $('query-input'),
  sendBtn: $('send-btn'),
};

let busy = false;

/* ------------------------------------------------------------------ 工具 */

function showBanner(text, kind) {
  el.banner.textContent = text;
  el.banner.className = `banner ${kind || 'warn'}`;
}

function hideBanner() {
  el.banner.className = 'banner hidden';
}

function setStatus(text, kind) {
  el.statusText.textContent = text;
  el.statusDot.className = `dot ${kind || ''}`;
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

function hideEmptyState() {
  if (el.emptyState && el.emptyState.parentNode) {
    el.emptyState.remove();
  }
}

/* --------------------------------------------------------- 消息渲染 */

function addUserMessage(text) {
  hideEmptyState();
  const wrap = document.createElement('div');
  wrap.className = 'msg user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  el.messages.appendChild(wrap);
  scrollToBottom();
}

function addTyping() {
  hideEmptyState();
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  wrap.id = 'typing-msg';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  // 生成回答需要几秒，给出文字提示，避免用户以为卡住
  bubble.innerHTML =
    '<span class="typing"><i></i><i></i><i></i></span>' +
    '<span class="typing-hint">正在检索并组织回答…</span>';
  wrap.appendChild(bubble);
  el.messages.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function removeTyping() {
  const t = $('typing-msg');
  if (t) t.remove();
}

/** 渲染机器人回复：命中列表或未找到提示 */
function addBotMessage(payload) {
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (!payload.matched) {
    const p = document.createElement('div');
    p.className = 'no-hit';
    p.textContent = '未在知识库中找到相关内容。可以换个说法再试，或补充相关知识点后重建索引。';
    bubble.appendChild(p);
  } else {
    // 优先展示大模型生成的回答；没有生成时（未启用/只检索）退回材料列表
    if (payload.answer) {
      const ans = document.createElement('div');
      ans.className = 'answer-text';
      ans.textContent = payload.answer;
      bubble.appendChild(ans);

      if (payload.generation_error) {
        const warn = document.createElement('div');
        warn.className = 'gen-warn';
        warn.textContent = '回答生成失败，以下为检索到的原始内容。';
        bubble.appendChild(warn);
      }
    }

    // 依据知识条目折叠展示：默认收起，避免长列表淹没回答
    if (payload.hits && payload.hits.length) {
      const details = document.createElement('details');
      details.className = 'sources';

      const summary = document.createElement('summary');
      summary.textContent = `依据（${payload.hits.length} 条知识）`;
      details.appendChild(summary);

      payload.hits.forEach((hit) => {
        const card = document.createElement('div');
        card.className = 'hit';

        const head = document.createElement('div');
        head.className = 'hit-head';

        const q = document.createElement('div');
        q.className = 'hit-q';
        q.textContent = hit.question;
        head.appendChild(q);

        // 匹配度：rerank 开启时展示 rerank 分数，否则展示向量相似度
        const score = hit.rerank_score > 0 ? hit.rerank_score : hit.score;
        const label = hit.rerank_score > 0 ? '精排' : '相似度';
        const bScore = document.createElement('span');
        bScore.className = 'badge';
        bScore.textContent = `${label} ${score.toFixed(3)}`;
        head.appendChild(bScore);

        if (hit.variant_count > 1) {
          const bVar = document.createElement('span');
          bVar.className = 'badge src';
          bVar.textContent = `${hit.variant_count} 个问法命中`;
          head.appendChild(bVar);
        }

        card.appendChild(head);

        const a = document.createElement('div');
        a.className = 'hit-a';
        a.textContent = hit.answer;
        card.appendChild(a);

        details.appendChild(card);
      });

      bubble.appendChild(details);

      // 只有依据、没有生成回答时（LLM 未启用），默认展开，否则界面上什么都看不到
      if (!payload.answer) details.open = true;
    }
  }

  wrap.appendChild(bubble);
  el.messages.appendChild(wrap);
  scrollToBottom();
}

function addErrorMessage(text) {
  removeTyping();
  const wrap = document.createElement('div');
  wrap.className = 'msg bot';
  const bubble = document.createElement('div');
  bubble.className = 'bubble error';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  el.messages.appendChild(wrap);
  scrollToBottom();
}

/* ------------------------------------------------------------ 请求 */

async function ask(query) {
  if (busy) return;
  busy = true;
  el.sendBtn.disabled = true;

  addUserMessage(query);
  el.input.value = '';
  addTyping();

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    removeTyping();

    if (!resp.ok) {
      let detail = `请求失败（HTTP ${resp.status}）`;
      try {
        const j = await resp.json();
        if (j && j.detail) detail = j.detail;
      } catch (_) { /* 响应不是 json，用默认文案 */ }
      addErrorMessage(detail);
      return;
    }

    addBotMessage(await resp.json());
  } catch (err) {
    removeTyping();
    addErrorMessage(`无法连接服务：${err.message}。请确认服务已启动。`);
    setStatus('连接失败', 'err');
  } finally {
    busy = false;
    el.sendBtn.disabled = false;
    el.input.focus();
  }
}

/* -------------------------------------------------------- 初始化 */

function initMapButton(mapUrl) {
  if (mapUrl) {
    el.mapBtn.href = mapUrl;
    el.mapBtn.classList.remove('disabled');
    el.mapBtn.title = `打开：${mapUrl}`;
  } else {
    // 未配置跳转链接：按钮可见但置灰，点击给出明确提示，避免用户以为是坏的
    el.mapBtn.href = '#';
    el.mapBtn.classList.add('disabled');
    el.mapBtn.title = '尚未配置跳转链接';
    el.mapBtn.addEventListener('click', (e) => {
      e.preventDefault();
      showBanner('「查询地图」的跳转链接尚未配置。在 .env 中设置 QUERY_MAP_URL 后重启服务即可生效。', 'warn');
      setTimeout(hideBanner, 6000);
    });
  }
}

function initSuggestions(list) {
  el.suggestions.innerHTML = '';
  list.forEach((text) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = text;
    b.addEventListener('click', () => {
      el.input.value = text;
      el.form.requestSubmit();
    });
    el.suggestions.appendChild(b);
  });
}

async function loadUiConfig() {
  try {
    const r = await fetch('/ui-config');
    if (!r.ok) throw new Error(String(r.status));
    const cfg = await r.json();

    if (cfg.title) {
      el.title.textContent = cfg.title;
      document.title = cfg.title;
    }
    initMapButton(cfg.map_url);
  } catch (_) {
    // 配置接口拿不到不影响提问，只是地图按钮保持未配置态
    initMapButton('');
  }

  // 推荐问题：直接取自知识库更贴切，但前端不直连知识库，
  // 这里给通用引导，避免写死与知识库不一致的内容。
  initSuggestions(['你是谁', '学校在哪', '宿舍条件怎么样', '校训是什么']);
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (!r.ok) throw new Error(String(r.status));
    const h = await r.json();

    if (h.ready) {
      setStatus('就绪', 'ok');
      hideBanner();
    } else {
      setStatus('索引未就绪', 'warn');
      showBanner(h.hint || '索引尚未构建，请先运行 build_index.py 再重启服务。', 'warn');
    }
  } catch (_) {
    setStatus('无法连接', 'err');
  }
}

function init() {
  el.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = el.input.value.trim();
    if (q) ask(q);
  });

  el.statusBtn.addEventListener('click', checkHealth);

  loadUiConfig();
  checkHealth();
  // 索引可能在建好后才启动服务，30 秒轮询一次即可，避免无谓请求
  setInterval(checkHealth, 30000);

  el.input.focus();
}

document.addEventListener('DOMContentLoaded', init);
