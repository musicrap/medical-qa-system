import { defineStore } from "pinia";
import { ref } from "vue";
import { chatQuery, chatStream, getKnowledgeStatus, importKnowledge, loadKnowledge } from "../api";

export const useChatStore = defineStore("chat", () => {
  const messages = ref([]);
  const agentTrace = ref([]);
  const sessionId = ref(crypto.randomUUID());
  const isThinking = ref(false);
  const kbStatus = ref({ total_documents: 0, status: "unknown" });

  async function sendMessageStream(query) {
    messages.value.push({ role: "user", content: query });
    isThinking.value = true;

    // ????? token ??????????????
    let assistantMsg = null;
    let msgIndex = -1;

    try {
      for await (const event of chatStream(query, sessionId.value)) {
        if (event.type === 'token') {
          if (!assistantMsg) {
            assistantMsg = { role: "assistant", content: "", sources: [] };
            messages.value.push(assistantMsg);
            isThinking.value = false;  // ??????
            msgIndex = messages.value.length - 1;
          }
          assistantMsg.content += event.data;
          messages.value[msgIndex] = { ...assistantMsg };
        } else if (event.type === 'meta') {
          if (!assistantMsg) {
            assistantMsg = { role: "assistant", content: "", sources: [] };
            messages.value.push(assistantMsg);
            msgIndex = messages.value.length - 1;
          }
          assistantMsg.sources = event.data.sources || [];
          agentTrace.value = event.data.agent_trace || [];
          if (event.data.from_cache) {
            agentTrace.value = ["[Memory] 已从历史缓存中获取答案"];
          }
          messages.value[msgIndex] = { ...assistantMsg };
        }
      }
      // ????????
      if (!assistantMsg) {
        assistantMsg = { role: "assistant", content: "??????????", sources: [] };
        messages.value.push(assistantMsg);
      }
    } catch (e) {
      if (!assistantMsg) {
        assistantMsg = { role: "assistant", content: "", sources: [] };
        messages.value.push(assistantMsg);
        msgIndex = messages.value.length - 1;
      }
      assistantMsg.content = `❌ 请求失败: ${e.message}`;
      messages.value[msgIndex] = { ...assistantMsg };
    } finally {
      isThinking.value = false;
    }
  }
  async function sendMessage(query) {
    messages.value.push({ role: "user", content: query });
    isThinking.value = true;

    try {
      const data = await chatQuery(query, sessionId.value);
      messages.value.push({
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
      });
      agentTrace.value = data.agent_trace || [];
    } catch (e) {
      messages.value.push({
        role: "assistant",
        content: `❌ 请求失败: ${e.message}`,
        sources: [],
      });
    } finally {
      isThinking.value = false;
    }
  }

  async function loadKBStatus() {
    try {
      const data = await getKnowledgeStatus();
      kbStatus.value = data;
    } catch (e) {
      kbStatus.value = { total_documents: 0, status: "error" };
    }
  }

  async function loadKB() {
    const data = await loadKnowledge();
    kbStatus.value = { total_documents: data.imported_count, status: "ready" };
    return data;
  }

  async function buildKB() {
    const data = await importKnowledge(true);
    kbStatus.value = { total_documents: data.imported_count, status: "ready" };
    return data;
  }

  function newSession() {
    sessionId.value = crypto.randomUUID();
    messages.value = [];
    agentTrace.value = [];
  }

  return { messages, agentTrace, sessionId, isThinking, kbStatus, sendMessage, sendMessageStream, loadKBStatus, loadKB, buildKB, newSession };
});
