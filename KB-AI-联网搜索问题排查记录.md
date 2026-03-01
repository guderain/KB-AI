# KB-AI 联网搜索问题排查记录

## 背景
目标：当本地知识库未命中时，KB-AI 自动联网搜索，而不是直接返回“我不知道”。

测试问题：`鳗鱼饭怎么做`

---

## 1. 现象一：始终返回“我不知道”

### 表现
- `kb-chat` / `kb-chat-stream` 对同一问题持续返回“我不知道”。

### 根因
1. 未安装 `duckduckgo-search`，SDK 路径不可用。
2. 网络请求超时（`httpx.ConnectTimeout`），HTML 兜底也失败。
3. 旧结果被 Redis 缓存命中（`cache_hit=True`），导致后续不再重新检索。

### 修复
- 增加无 SDK 的 HTML 搜索兜底。
- 增加联网配置项：
  - `WEB_SEARCH_TIMEOUT_SECONDS`
  - `WEB_SEARCH_TRUST_ENV`
- 修改缓存策略：未知答案（“我不知道”语义）不写缓存。
- 清空 Redis + 重启 KB-AI。

---

## 2. 现象二：`pip install duckduckgo-search` 安装失败

### 表现
- `No matching distribution found for duckduckgo-search`
- `Cannot connect to proxy (WinError 10061)`

### 根因
1. 会话环境存在 `PIP_NO_INDEX=1`（离线模式）。
2. 代理配置不可用，导致 pip 无法访问索引。
3. 一度出现系统临时目录权限问题（Temp 目录写入受限）。

### 修复
- 取消离线模式并清理代理环境变量。
- 由于网络环境不稳定，最终采用“无需该 SDK 也能检索”的代码方案作为主兜底。

---

## 3. 现象三：联网结果不相关（微软页面）

### 表现
- 问题是“鳗鱼饭怎么做”，返回来源却是 `microsoft.com` 等无关页面。

### 根因
- 旧逻辑只要某 provider 有结果就直接采用，缺少“结果与问题相关性”校验。
- 因此不会继续回退到后续 provider（Tavily/百度）。

### 修复
- 增加相关性评分与过滤（基于 query token 与标题/摘要匹配）。
- 当某 provider 结果低相关时，自动回退下一个 provider。
- provider 异常改为“单源失败不中断全链路”。

---

## 4. 搜索源策略调整

### 目标顺序
- `Bing -> Tavily -> Baidu`（`auto` 模式）

### 实施
- 已在代码中实现该顺序。
- 增加配置：
  - `WEB_SEARCH_PROVIDER=auto`
  - `TAVILY_API_KEY=...`

---

## 5. 现象四：Tavily 返回 400（Query is invalid）

### 表现
- Tavily 请求返回：`{"detail":{"error":"Query is invalid."}}`

### 根因
- 本地终端/编码调试过程中，中文 query 被错误传成 `????`，导致 Tavily 判定无效。
- 验证后确认 Tavily 对中文 query 本身是可用的（直连返回 `200`）。

### 修复
- Tavily 请求改为直接使用原始 query（不做中间错误转换）。
- 继续保留 `WEB_SEARCH_TRUST_ENV=false`，避免坏代理干扰。

---

## 6. 连通性结论（本次环境）

- `trust_env=True`：易受系统代理影响，出现连接拒绝（10061）。
- `trust_env=False`：可直连 Tavily（可达，状态 200，能返回 results）。

建议：本机调试阶段优先使用：

```env
WEB_SEARCH_PROVIDER=auto
WEB_SEARCH_TRUST_ENV=false
WEB_SEARCH_TIMEOUT_SECONDS=12
TAVILY_API_KEY=你的有效key
```

---

## 7. 最终状态

- 已完成：
  1. Redis 清理
  2. KB-AI 重启
  3. 健康检查 `200`
  4. 搜索链路代码修复（回退、相关性过滤、Tavily可用性）

- 当前建议：
  - 持续观察 `kb-chat` 的 `sources` 是否与问题语义一致。
  - 若出现误检索，优先查看 provider 命中日志与相关性过滤阈值。

---

## 8. 相关变更点（代码）

- `D:\KB-AI\app\services\rag_service_fast.py`
- `D:\KB-AI\app\core\config.py`
- `D:\KB-AI\.env`

