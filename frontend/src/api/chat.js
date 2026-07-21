import request from "./request";

// RAG问答（非流式，作为降级方案）
export const askQuestion = (query, sessionId) => {
  return request.post("/ask", {
    query,
    session_id: sessionId || null
  });
};

/**
 * SSE 流式问答：fetch + ReadableStream 消费 text/event-stream。
 * axios 不支持浏览器流式响应，这里直接用 fetch。
 *
 * @param {string} query 用户问题
 * @param {string} sessionId 会话ID（多轮对话）
 * @param {object} callbacks { onToken(text), onSources(sources), onError(err) }
 * @returns {Promise<void>} 流结束后 resolve
 */
export const askQuestionStream = async (query, sessionId, { onToken, onSources, onError } = {}) => {
  const token = localStorage.getItem("token");

  const resp = await fetch("/api/ask/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify({
      query,
      session_id: sessionId || null
    })
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`流式请求失败: HTTP ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  // SSE 按 "\n\n" 切分事件，每条以 "data: " 开头
  const handleEvent = (raw) => {
    const data = raw.trim();
    if (!data.startsWith("data:")) return;
    const payload = data.slice(5).trim();
    if (!payload || payload === "[DONE]") return;

    try {
      const event = JSON.parse(payload);
      if (event.type === "token") {
        onToken?.(event.text || "");
      } else if (event.type === "end") {
        onSources?.(event.sources || []);
      } else if (event.type === "error") {
        onError?.(new Error(event.detail || "服务端流式错误"));
      }
    } catch (e) {
      console.warn("SSE 事件解析失败:", payload, e);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // 最后一段可能不完整，留给下次
    parts.forEach(handleEvent);
  }

  if (buffer.trim()) {
    handleEvent(buffer);
  }
};
