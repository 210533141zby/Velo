<script setup lang="ts">
import { ref } from 'vue';
import { Home, FileText, Search, Settings, Plus } from 'lucide-vue-next';

// 模拟数据 - 确保按钮有反馈
const docs = ref([
  { id: 1, title: '项目架构说明书', active: true },
  { id: 2, title: 'API 接口定义', active: false },
  { id: 3, title: '需求分析笔记', active: false },
]);

const handleCreate = () => {
  console.log("Create new doc triggered");
  docs.value.unshift({ id: Date.now(), title: 'Untitled Document', active: true });
};
</script>

<template>
  <div class="h-14 flex items-center px-4 border-b border-zinc-100">
    <div class="font-bold text-lg tracking-tight flex items-center gap-2">
      <div class="w-5 h-5 bg-black rounded-full"></div>
      Velo.
    </div>
  </div>

  <div class="px-4 py-4">
    <button class="w-full bg-white border border-zinc-200 text-zinc-400 rounded-lg h-9 flex items-center px-3 text-sm shadow-sm hover:border-zinc-300 hover:text-zinc-600 transition-all group">
      <Search class="w-4 h-4 mr-2 group-hover:scale-110 transition-transform" />
      <span>Search...</span>
      <kbd class="ml-auto font-mono text-xs bg-zinc-100 px-1.5 rounded text-zinc-500">⌘K</kbd>
    </button>
  </div>

  <nav class="flex-1 px-2 space-y-0.5 overflow-y-auto">
    <div class="px-2 mb-2 text-[10px] font-bold text-zinc-400 uppercase tracking-wider">Library</div>
    
    <a href="#" class="flex items-center gap-2 px-2 py-1.5 text-sm text-zinc-600 rounded-md hover:bg-zinc-100 transition-colors">
      <Home class="w-4 h-4" /> Home
    </a>
    
    <div class="mt-4 px-2 mb-2 text-[10px] font-bold text-zinc-400 uppercase tracking-wider flex justify-between items-center">
      <span>Documents</span>
      <button @click="handleCreate" class="hover:bg-zinc-200 rounded p-0.5 transition-colors">
        <Plus class="w-3 h-3 text-zinc-600" />
      </button>
    </div>

    <div v-for="doc in docs" :key="doc.id">
      <button 
        class="w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-md transition-colors text-left truncate"
        :class="doc.active ? 'bg-white shadow-sm text-black font-medium border border-zinc-100' : 'text-zinc-500 hover:bg-zinc-100 hover:text-zinc-900'"
      >
        <FileText class="w-3.5 h-3.5 opacity-70" />
        {{ doc.title }}
      </button>
    </div>
  </nav>

  <div class="p-4 border-t border-zinc-200">
    <button class="flex items-center gap-2 text-sm text-zinc-500 hover:text-black transition-colors w-full">
      <Settings class="w-4 h-4" /> Settings
    </button>
  </div>
</template>