<template>
  <div class="app-layout">
    <header class="app-header">
      <div class="header-left">
        <div class="logo-icon">🏥</div>
        <div class="logo-text">
          <span class="logo-title">医疗知识问答</span>
          <span class="logo-sub">Medical QA System</span>
        </div>
      </div>
      <nav class="header-nav">
        <router-link to="/" class="nav-link" :class="{ active: $route.path === '/' }">
          <el-icon><ChatDotRound /></el-icon>
          <span>智能问答</span>
        </router-link>
        <router-link to="/knowledge" class="nav-link" :class="{ active: $route.path === '/knowledge' }">
          <el-icon><FolderOpened /></el-icon>
          <span>知识库</span>
        </router-link>
      </nav>
      <div class="header-right">
        <span class="status-indicator" :class="healthStatus">
          <span class="status-pulse"></span>
          {{ healthStatus === 'healthy' ? '服务运行中' : '检查中...' }}
        </span>
      </div>
    </header>
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { healthCheck } from './api'

const healthStatus = ref('unknown')

onMounted(async () => {
  try {
    await healthCheck()
    healthStatus.value = 'healthy'
  } catch {
    healthStatus.value = 'unhealthy'
  }
})
</script>

<style scoped>
.app-layout {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  height: 60px;
  background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #059669 100%);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 2px 12px rgba(6,78,59,0.15);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  width: 40px;
  height: 40px;
  background: rgba(255,255,255,0.15);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}

.logo-text {
  display: flex;
  flex-direction: column;
}

.logo-title {
  font-size: 16px;
  font-weight: 700;
  color: white;
  line-height: 1.2;
}

.logo-sub {
  font-size: 10px;
  color: rgba(255,255,255,0.6);
  letter-spacing: 0.5px;
}

.header-nav {
  display: flex;
  gap: 4px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: 10px;
  color: rgba(255,255,255,0.75);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
}

.nav-link:hover {
  background: rgba(255,255,255,0.12);
  color: white;
}

.nav-link.active {
  background: rgba(255,255,255,0.18);
  color: white;
}

.header-right {
  display: flex;
  align-items: center;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: rgba(255,255,255,0.7);
  padding: 4px 12px;
  background: rgba(255,255,255,0.08);
  border-radius: 20px;
}

.status-pulse {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #10b981;
  animation: pulse 2s infinite;
}

.status-indicator.healthy .status-pulse {
  background: #34d399;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.app-main {
  flex: 1;
  background: #f8fafc;
}
</style>
