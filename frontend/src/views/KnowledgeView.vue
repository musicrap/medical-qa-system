<template>
  <div class="knowledge-view">
    <div class="kb-hero">
      <div class="hero-content">
        <div class="hero-icon">
          <span>🧬</span>
        </div>
        <div class="hero-text">
          <h2>医学知识库管理</h2>
          <p>管理和检索医疗知识向量库，为智能问答提供知识支撑</p>
        </div>
        <el-button
          :type="store.kbStatus.status === 'ready' ? 'success' : 'primary'"
          size="large"
          round
          @click="handleLoad"
          :loading="loading"
          :disabled="store.kbStatus.status === 'ready'"
          class="load-btn"
        >
          <el-icon><Download /></el-icon>
          {{ store.kbStatus.status === 'ready' ? '✓ 已加载' : '加载数据' }}
        </el-button>
      </div>
    </div>

    <transition name="progress-slide">
      <div v-if="loading" class="progress-card">
        <div class="progress-header">
          <span class="progress-stage">{{ progressMessage }}</span>
          <span class="progress-pct">{{ progress }}%</span>
        </div>
        <el-progress
          :percentage="progress"
          :status="progressStatus"
          :stroke-width="14"
          :text-inside="false"
          :color="progressColor"
        />
      </div>
    </transition>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon" style="background: linear-gradient(135deg, #ecfdf5, #d1fae5)">
          <span>📊</span>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ store.kbStatus.total_documents.toLocaleString() }}</div>
          <div class="stat-label">向量文档总数</div>
        </div>
        <el-tag :type="statusType" size="small" effect="plain" round>{{ statusText }}</el-tag>
      </div>

      <div class="stat-card">
        <div class="stat-icon" style="background: linear-gradient(135deg, #eff6ff, #dbeafe)">
          <span>🗂️</span>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ store.kbStatus.collection_name || 'medical_knowledge' }}</div>
          <div class="stat-label">集合名称</div>
        </div>
      </div>

      <div class="stat-card">
        <div class="stat-icon" style="background: linear-gradient(135deg, #fef3c7, #fde68a)">
          <span>📁</span>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ store.kbStatus.data_file || "train-sft.jsonl" }}</div>
          <div class="stat-label">数据文件</div>
        </div>
      </div>
    </div>

    <div class="search-section">
      <div class="search-header">
        <h3>🔍 知识库检索测试</h3>
        <span class="search-hint">验证已加载的医学知识是否可正常检索</span>
      </div>

      <div class="search-bar">
        <el-input
          v-model="searchQuery"
          placeholder="输入医学关键词测试知识库检索..."
          size="large"
          @keydown.enter="handleSearch"
          clearable
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
          <template #append>
            <el-button @click="handleSearch" :loading="searching" type="primary">
              搜索
            </el-button>
          </template>
        </el-input>
      </div>

      <transition-group name="result-list" tag="div" class="results-list">
        <div
          v-for="(result, idx) in searchResults"
          :key="idx"
          class="result-item"
        >
          <div class="result-rank">#{{ idx + 1 }}</div>
          <div class="result-body">
            <div class="result-meta">
              <el-tag size="small" effect="dark" round>相关度 {{ (result.score * 100).toFixed(1) }}%</el-tag>
            </div>
            <p class="result-text">{{ result.content?.substring(0, 300) }}</p>
          </div>
        </div>
      </transition-group>

      <el-empty
        v-if="searchQuery && !searching && searchResults.length === 0"
        description="未找到相关医学知识"
        :image-size="80"
      />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useChatStore } from '../stores/chat'
import { searchKnowledge } from '../api'
import { ElMessage } from 'element-plus'
import { Download, Search } from '@element-plus/icons-vue'

const store = useChatStore()
const loading = ref(false)
const searching = ref(false)
const searchQuery = ref('')
const searchResults = ref([])
const progress = ref(0)
const progressMessage = ref('')

const progressStatus = computed(() => {
  if (progress.value >= 100) return 'success'
  return ''
})

const progressColor = computed(() => {
  if (progress.value < 30) return '#10b981'
  if (progress.value < 70) return '#0ea5e9'
  return '#059669'
})

const statusType = computed(() =>
  store.kbStatus.status === 'ready' ? 'success' : 'warning'
)

const statusText = computed(() =>
  store.kbStatus.status === 'ready' ? '已加载' : '未加载'
)

async function handleLoad() {
  loading.value = true
  progress.value = 0
  progressMessage.value = '正在连接服务...'

  try {
    const response = await fetch('http://localhost:8000/api/knowledge/load', { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6))
            if (event.error) throw new Error(event.error)
            progress.value = event.progress || 0
            progressMessage.value = event.message || ''
            if (event.stage === 'done' || event.stage === 'skipped') {
              store.kbStatus = {
                total_documents: event.imported_count || store.kbStatus.total_documents,
                status: 'ready',
                collection_name: 'medical_knowledge'
              }
              ElMessage.success(event.stage === 'skipped'
                ? `知识库已有数据，无需重复加载`
                : `✓ 加载完成！共 ${event.imported_count.toLocaleString()} 条向量记录`)
            }
          } catch (parseErr) {
            // skip
          }
        }
      }
    }
  } catch (e) {
    ElMessage.error(`加载失败: ${e.message}`)
    progress.value = 0
  } finally {
    loading.value = false
  }
}

async function handleSearch() {
  if (!searchQuery.value.trim()) return
  searching.value = true
  try {
    const data = await searchKnowledge(searchQuery.value, 5)
    searchResults.value = data.results || []
    if (data.count) ElMessage.success(`找到 ${data.count} 条相关结果`)
    else ElMessage.info('未找到相关结果')
  } catch (e) {
    ElMessage.error(`搜索失败: ${e.message}`)
  } finally {
    searching.value = false
  }
}

onMounted(() => {
  store.loadKBStatus()
})
</script>

<style scoped>
.knowledge-view {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px 20px 48px;
}

.kb-hero {
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 50%, #a7f3d0 100%);
  border-radius: 20px;
  padding: 32px;
  margin-bottom: 20px;
  position: relative;
  overflow: hidden;
}

.kb-hero::after {
  content: '';
  position: absolute;
  right: -40px;
  top: -40px;
  width: 200px;
  height: 200px;
  background: rgba(16, 185, 129, 0.08);
  border-radius: 50%;
}

.hero-content {
  display: flex;
  align-items: center;
  gap: 20px;
  position: relative;
  z-index: 1;
}

.hero-icon {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #10b981, #059669);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  box-shadow: 0 4px 16px rgba(16, 185, 129, 0.3);
}

.hero-text {
  flex: 1;
}

.hero-text h2 {
  font-size: 22px;
  font-weight: 700;
  color: #064e3b;
  margin: 0 0 6px;
}

.hero-text p {
  font-size: 14px;
  color: #047857;
  margin: 0;
}

.load-btn {
  font-weight: 600;
  box-shadow: 0 2px 12px rgba(16, 185, 129, 0.25);
}

.progress-card {
  background: white;
  border-radius: 14px;
  padding: 20px 24px;
  margin-bottom: 20px;
  box-shadow: 0 1px 8px rgba(0,0,0,0.04);
  border: 1px solid #e5e7eb;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.progress-stage {
  font-size: 14px;
  color: #374151;
  font-weight: 500;
}

.progress-pct {
  font-size: 14px;
  color: #059669;
  font-weight: 700;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-card {
  background: white;
  border-radius: 14px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 14px;
  position: relative;
  box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  border: 1px solid #f1f5f9;
  transition: box-shadow 0.2s;
}

.stat-card:hover {
  box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}

.stat-card > .el-tag {
  position: absolute;
  top: 12px;
  right: 14px;
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
}

.stat-value {
  font-size: 20px;
  font-weight: 700;
  color: #1f2937;
  line-height: 1.3;
}

.stat-label {
  font-size: 12px;
  color: #9ca3af;
}

.search-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.03);
  border: 1px solid #f1f5f9;
}

.search-header {
  margin-bottom: 16px;
}

.search-header h3 {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 4px;
}

.search-hint {
  font-size: 13px;
  color: #9ca3af;
}

.search-bar {
  margin-bottom: 20px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-item {
  display: flex;
  gap: 14px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  transition: all 0.2s;
}

.result-item:hover {
  background: #f0fdf4;
  border-color: #bbf7d0;
}

.result-rank {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #10b981, #059669);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}

.result-body {
  flex: 1;
}

.result-meta {
  margin-bottom: 8px;
}

.result-text {
  font-size: 14px;
  color: #4b5563;
  line-height: 1.7;
  margin: 0;
}

.progress-slide-enter-active {
  transition: all 0.4s ease;
}

.progress-slide-leave-active {
  transition: all 0.3s ease;
}

.progress-slide-enter-from {
  opacity: 0;
  transform: translateY(-10px);
}

.progress-slide-leave-to {
  opacity: 0;
  transform: translateY(-10px);
}

.result-list-enter-active {
  transition: all 0.4s ease;
}

.result-list-leave-active {
  transition: all 0.2s ease;
}

.result-list-enter-from {
  opacity: 0;
  transform: translateX(-20px);
}

.result-list-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .stats-grid {
    grid-template-columns: 1fr;
  }
  .hero-content {
    flex-direction: column;
    text-align: center;
  }
}
</style>
