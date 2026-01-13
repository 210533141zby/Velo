# 前端加载死循环与 OpenAI 连接超时故障报告

## 1. 问题描述

### 1.1 现象
用户反馈主要包含两个核心问题：
1.  **主界面一直处于加载状态**：打开网页后，页面显示空白或一直转圈，无法进入文档库列表。
2.  **OpenAI 连接超时**：后端日志显示 `openai.APITimeoutError`，导致文档索引或 RAG 功能失败，甚至引发服务不稳定。

### 1.2 影响范围
-   **前端可用性**：完全不可用（Critical）。用户无法查看、编辑或操作任何文档。
-   **后端稳定性**：部分功能受损（Major）。文档创建/更新时的向量化索引失败，导致搜索和问答功能失效。

---

## 2. 故障排查过程

### 2.1 前端加载死循环分析
通过代码审查 `backend/static/js/app.js`，发现存在语法错误导致 Vue 应用无法挂载。

-   **错误定位**：在 `setup()` 函数内部，变量 `formatDate` 被声明了两次。
-   **代码片段**：
    ```javascript
    // 第 203 行附近
    const formatDate = (dateStr) => { ... }

    // 第 440 行附近
    const formatDate = (dateString) => { ... }
    ```
-   **原因**：JavaScript 不允许在同一作用域内使用 `const` 重复声明同名变量。这导致脚本抛出 `SyntaxError: Identifier 'formatDate' has already been declared`。
-   **后果**：由于脚本执行中断，Vue 实例从未创建，`index.html` 中的 `v-cloak` 样式（用于隐藏未编译模板）一直生效，导致页面看起来像是“一直在加载”或空白。

### 2.2 OpenAI 连接超时分析
后端日志显示在调用 `_get_len_safe_embeddings` 时发生超时。

-   **错误日志**：
    ```
    openai.APITimeoutError: Request timed out.
    File ".../langchain_openai/embeddings/base.py", line 576
    ```
-   **原因**：LangChain 在进行文档向量化（Embedding）时，默认通过 OpenAI API 进行网络请求。由于网络波动或 API 响应延迟，请求超过了默认超时时间。
-   **风险**：如果在主线程或未捕获异常的后台线程中发生此错误，可能导致服务崩溃或任务队列阻塞。

---

## 3. 解决方案

### 3.1 修复前端语法错误
-   **操作**：删除了 `app.js` 中第 203 行附近旧的、功能较弱的 `formatDate` 函数，保留了第 440 行支持“时:分:秒”格式的新版本。
-   **结果**：消除了语法错误，Vue 应用可以正常启动和挂载，界面恢复显示。

### 3.2 增强后端健壮性 (Robustness)
-   **操作**：在 `backend/app/service.py` 的 `index_document` 方法中增加了重试机制和异常处理。
-   **代码变更**：
    ```python
    # 引入 APITimeoutError
    from openai import APITimeoutError
    
    # 增加重试循环 (Max 3次)
    for retry in range(max_retries):
        try:
            await run_in_threadpool(self.vector_store.add_documents, docs)
            break
        except (APITimeoutError, Exception) as e:
            # 指数退避策略 (2s, 4s...)
            await asyncio.sleep(retry_delay)
    ```
-   **效果**：当 OpenAI API 暂时不可用或超时时，系统会自动重试，而不是直接报错失败。即使最终失败，也会记录详细日志并保证服务主进程不崩溃。

---

## 4. 验证与后续
1.  **验证**：重启服务后，刷新浏览器，前端页面应能瞬间加载。后端日志中若出现超时，应能看到“正在重试”的警告日志而非直接报错。
2.  **建议**：建议检查网络环境配置（如代理设置），确保后端服务器能稳定访问 OpenAI API 域名。
