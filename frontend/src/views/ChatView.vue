<template>
  <div class="chat-view">
    <div class="chat-layout">
      <aside class="sidebar" v-if="showSidebar">
        <div class="sidebar-brand">
          <span class="brand-icon">🏥</span>
          <div>
            <div class="brand-name">医疗知识库</div>
            <div class="brand-status">
              <span class="status-dot" :class="store.kbStatus.status"></span>
              {{ store.kbStatus.status === 'ready' ? '就绪' : '未加载' }}
            </div>
          </div>
        </div>

        <div class="sidebar-stats">
          <div class="stat-row">
            <span class="stat-label">向量文档</span>
            <span class="stat-num">{{ store.kbStatus.total_documents.toLocaleString() }}</span>
          </div>
          <div class="stat-row">
            <span class="stat-label">会话消息</span>
            <span class="stat-num">{{ store.messages.length }}</span>
          </div>
        </div>

        <div class="sidebar-section" v-if="store.agentTrace.length">
          <div class="section-title">🧠 Agent 执行轨迹</div>
          <div class="trace-list">
            <div class="trace-item" v-for="(trace, i) in store.agentTrace.slice(-6)" :key="i">
              <span class="trace-dot"></span>
              <span class="trace-text">{{ trace }}</span>
            </div>
          </div>
        </div>

        <div class="sidebar-actions">
          <el-button @click="store.newSession()" :icon="RefreshRight" round>
            新建会话
          </el-button>
        </div>

        <div class="sidebar-footer">
          <el-alert type="warning" :closable="false" show-icon>
            <template #title>
              <span style="font-size:12px">本系统仅供参考，不能替代专业医疗诊断</span>
            </template>
          </el-alert>
        </div>
      </aside>

      <div class="chat-main">
        <div class="chat-messages" ref="messagesContainer">
          <div v-if="store.messages.length === 0" class="welcome-area">
            <div class="welcome-card">
              <div class="welcome-icon">🏥</div>
              <h2>医疗知识问答助手</h2>
              <p>基于医学知识库的智能问答，为您提供专业参考</p>
              <div class="quick-questions">
                <div
                  v-for="q in quickQuestions"
                  :key="q"
                  @click="handleQuickQuestion(q)"
                  class="quick-chip"
                >
                  {{ q }}
                </div>
              </div>
            </div>
          </div>

          <div
            v-for="(msg, idx) in store.messages"
            :key="idx"
            :class="['msg-bubble', msg.role]"
          >
            <div class="bubble-avatar">
              {{ msg.role === 'user' ? '👤' : '🏥' }}
            </div>
            <div class="bubble-body">
              <div class="bubble-content">
                <div v-html="renderMarkdown(msg.content)" class="markdown-body"></div>
              </div>
              <div v-if="msg.sources && msg.sources.length" class="sources-block">
                <div class="sources-title">📖 参考来源</div>
                <div v-for="(src, si) in msg.sources.slice(0, 3)" :key="si" class="source-chip">
                  <span class="source-score">{{ (src.score * 100).toFixed(0) }}%</span>
                  <span class="source-text">{{ src.content?.substring(0, 120) }}...</span>
                </div>
              </div>
            </div>
          </div>

          <div v-if="store.isThinking" class="msg-bubble assistant">
            <div class="bubble-avatar">🏥</div>
            <div class="thinking-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>

        <div class="input-area">
          <div class="input-row">
            <el-input
              v-model="inputText"
              type="textarea"
              :rows="2"
              placeholder="输入您想咨询的医疗健康问题..."
              @keydown.enter.exact.prevent="handleSend"
              :disabled="store.isThinking"
              resize="none"
              class="chat-input"
            />
            <el-button
              type="primary"
              @click="handleSend"
              :loading="store.isThinking"
              :disabled="!inputText.trim()"
              :icon="Promotion"
              round
              class="send-btn"
            >
              发送
            </el-button>
          </div>
          <div class="input-hint">Enter 发送 · Shift+Enter 换行</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { useChatStore } from '../stores/chat'
import { marked } from 'marked'
import { Promotion, RefreshRight } from '@element-plus/icons-vue'

const store = useChatStore()
const inputText = ref('')
const messagesContainer = ref(null)
const showSidebar = ref(true)

const quickQuestions = [
  '高血压患者日常饮食需要注意什么？',
  '糖尿病的早期症状有哪些？',
  '如何科学预防流感？',
  '儿童发热的正确处理方法？',
]

function renderMarkdown(text) {
  if (!text) return ''
  return marked.parse(text)
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || store.isThinking) return
  inputText.value = ''
  await store.sendMessageStream(text)
  await nextTick()
  scrollToBottom()
}

function handleQuickQuestion(q) {
  inputText.value = q
  handleSend()
}

function scrollToBottom() {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

watch(() => store.messages.length, () => {
  nextTick(() => scrollToBottom())
})
// ????????????????
watch(() => {
  const msgs = store.messages
  return msgs.length ? msgs[msgs.length - 1].content : ''
}, () => {
  nextTick(() => scrollToBottom())
})

onMounted(() => {
  store.loadKBStatus()
})
</script>

<style scoped>
.chat-view {
  height: calc(100vh - 60px);
}

.chat-layout {
  display: flex;
  height: 100%;
}

.sidebar {
  width: 280px;
  background: white;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  padding: 20px;
  gap: 18px;
  flex-shrink: 0;
}

.sidebar-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f1f5f9;
}

.brand-icon {
  font-size: 28px;
  width: 44px;
  height: 44px;
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.brand-name {
  font-size: 15px;
  font-weight: 600;
  color: #1f2937;
}

.brand-status {
  font-size: 12px;
  color: #6b7280;
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #f59e0b;
}

.status-dot.ready {
  background: #10b981;
}

.sidebar-stats {
  background: #f8fafc;
  border-radius: 12px;
  padding: 14px;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
}

.stat-row + .stat-row {
  border-top: 1px solid #e5e7eb;
}

.stat-label {
  font-size: 13px;
  color: #6b7280;
}

.stat-num {
  font-size: 15px;
  font-weight: 700;
  color: #1f2937;
}

.section-title {
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 8px;
}

.trace-list {
  max-height: 180px;
  overflow-y: auto;
}

.trace-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 0;
  font-size: 12px;
  color: #6b7280;
}

.trace-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: #10b981;
  margin-top: 6px;
  flex-shrink: 0;
}

.sidebar-actions {
  margin-top: auto;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.welcome-area {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.welcome-card {
  text-align: center;
  max-width: 480px;
}

.welcome-icon {
  font-size: 56px;
  margin-bottom: 16px;
}

.welcome-card h2 {
  font-size: 24px;
  font-weight: 700;
  color: #1f2937;
  margin: 0 0 8px;
}

.welcome-card p {
  font-size: 14px;
  color: #6b7280;
  margin: 0 0 24px;
}

.quick-questions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
}

.quick-chip {
  padding: 10px 18px;
  background: white;
  border: 1px solid #d1fae5;
  border-radius: 20px;
  font-size: 13px;
  color: #047857;
  cursor: pointer;
  transition: all 0.2s;
}

.quick-chip:hover {
  background: #ecfdf5;
  border-color: #10b981;
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(16,185,129,0.12);
}

.msg-bubble {
  display: flex;
  gap: 10px;
  max-width: 78%;
  animation: msgIn 0.35s ease;
}

.msg-bubble.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.msg-bubble.assistant {
  align-self: flex-start;
}

.bubble-avatar {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.msg-bubble.user .bubble-avatar {
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
}

.msg-bubble.assistant .bubble-avatar {
  background: linear-gradient(135deg, #10b981, #059669);
}

.bubble-body {
  min-width: 0;
}

.bubble-content {
  padding: 14px 18px;
  border-radius: 18px;
  line-height: 1.75;
  word-break: break-word;
  font-size: 14px;
}

.msg-bubble.user .bubble-content {
  background: linear-gradient(135deg, #0ea5e9, #0284c7);
  color: white;
  border-bottom-right-radius: 6px;
}

.msg-bubble.assistant .bubble-content {
  background: white;
  border: 1px solid #e5e7eb;
  border-bottom-left-radius: 6px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.sources-block {
  margin-top: 8px;
  padding: 10px 14px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 12px;
}

.sources-title {
  font-size: 12px;
  font-weight: 600;
  color: #047857;
  margin-bottom: 6px;
}

.source-chip {
  display: flex;
  gap: 8px;
  padding: 6px 0;
  font-size: 12px;
  color: #4b5563;
  border-bottom: 1px dashed #d1fae5;
}

.source-chip:last-child {
  border-bottom: none;
}

.source-score {
  color: #059669;
  font-weight: 700;
  flex-shrink: 0;
}

.source-text {
  color: #6b7280;
}

.thinking-dots {
  display: flex;
  gap: 5px;
  padding: 16px 20px;
  background: white;
  border-radius: 18px;
  border: 1px solid #e5e7eb;
}

.thinking-dots span {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #10b981;
  animation: dotPulse 1.4s infinite ease-in-out;
}

.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

.input-area {
  padding: 16px 24px 20px;
  background: white;
  border-top: 1px solid #e5e7eb;
}

.input-row {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
}

.send-btn {
  background: linear-gradient(135deg, #10b981, #059669);
  border: none;
  box-shadow: 0 2px 8px rgba(16,185,129,0.3);
}

.send-btn:hover {
  background: linear-gradient(135deg, #059669, #047857);
}

.input-hint {
  font-size: 11px;
  color: #9ca3af;
  margin-top: 6px;
  text-align: center;
}

.markdown-body h1, .markdown-body h2, .markdown-body h3 {
  margin: 8px 0 4px;
  font-size: 1.05em;
}
.markdown-body p { margin: 4px 0; }
.markdown-body code {
  background: #f1f5f9;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
}
.markdown-body pre {
  background: #1e293b;
  color: #e2e8f0;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
}

@keyframes msgIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes dotPulse {
  0%, 60%, 100% { transform: translateY(0); opacity: 0.3; }
  30% { transform: translateY(-6px); opacity: 1; }
}

@media (max-width: 768px) {
  .sidebar { display: none; }
  .msg-bubble { max-width: 90%; }
}
</style>
