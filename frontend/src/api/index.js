import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  timeout: 300000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || "请求失败";
    return Promise.reject(new Error(message));
  }
);

export function chatQuery(query, sessionId) {
  return api.post("/chat", { query, session_id: sessionId, stream: false });
}

export function getKnowledgeStatus() {
  return api.get("/knowledge/status");
}

export function importKnowledge(reset = false) {
  return api.post("/knowledge/import", { reset });
}


export function loadKnowledge() {
  return api.post("/knowledge/load");
}
export function searchKnowledge(query, topK = 5) {
  return api.get("/knowledge/search", { params: { query, top_k: topK } });
}

export async function* chatStream(query, sessionId) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId, stream: true }),
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;
        if (data.startsWith('__META__')) {
          try {
            yield { type: 'meta', data: JSON.parse(data.slice(8)) };
          } catch (e) {
            // ignore parse errors
          }
        } else {
          yield { type: 'token', data: data.replace(/__NL__/g, '\n').replace(/__CR__/g, '\r') };
        }
      }
    }
  }
}

export function healthCheck() {
  return api.get("/health");
}

export default api;

