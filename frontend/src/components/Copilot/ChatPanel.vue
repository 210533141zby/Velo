<script setup lang="ts">
import { ref } from 'vue';
import { useEditorStore } from '../../stores/editorStore';
import { X, Send, Bot, User as UserIcon, Database } from 'lucide-vue-next';

const store = useEditorStore();
const inputValue = ref('');
const isRAGMode = ref(false); // Default to AI Chat, toggle to RAG (Database)

const messages = ref([
  { id: 1, role: 'ai', content: '你好！我是 AI 助手。今天有什么我可以帮你的吗？' }
]);

const sendMessage = () => {
  if (!inputValue.value.trim()) return;
  
  // Add User Message
  messages.value.push({
    id: Date.now(),
    role: 'user',
    content: inputValue.value
  });
  
  const userText = inputValue.value;
  const currentMode = isRAGMode.value ? 'RAG' : 'AI';
  inputValue.value = '';

  // Simulate AI Response (Placeholder for backend integration)
  setTimeout(() => {
    messages.value.push({
      id: Date.now() + 1,
      role: 'ai',
      content: `[${currentMode} 模式] 我收到了你的消息: "${userText}". 这是一个模拟回复。`
    });
  }, 1000);
};
</script>

<template>
  <div class="h-full flex flex-col bg-[#F5F5F5] border-l border-stone-200 shadow-xl font-sans">
    <!-- Header -->
    <div class="h-14 flex items-center justify-between px-4 border-b border-stone-200 bg-white z-10">
      <div class="flex items-center space-x-2">
        <span class="font-bold text-stone-700">AI 助手</span>
      </div>
      
      <button 
        @click="store.toggleCopilot"
        class="text-stone-400 hover:text-[#D06847] transition-colors p-1 rounded-md hover:bg-[#D06847]/10"
      >
        <X class="w-5 h-5" />
      </button>
    </div>

    <!-- Messages -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4 bg-[#F5F5F5]">
      <div 
        v-for="msg in messages" 
        :key="msg.id"
        class="flex items-start space-x-3"
        :class="msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''"
      >
        <div 
          class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 border border-stone-200 shadow-sm"
          :class="msg.role === 'ai' ? 'bg-white text-[#D06847]' : 'bg-stone-200 text-stone-600'"
        >
          <Database v-if="msg.role === 'ai' && isRAGMode" class="w-4 h-4" />
          <Bot v-else-if="msg.role === 'ai'" class="w-4 h-4" />
          <UserIcon v-else class="w-4 h-4" />
        </div>
        
        <div 
          class="max-w-[85%] rounded-lg p-3 text-sm leading-relaxed shadow-sm border"
          :class="msg.role === 'ai' 
            ? 'bg-white border-stone-200 text-stone-700' 
            : 'bg-[#D06847] text-white border-[#D06847]'"
        >
          {{ msg.content }}
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="p-4 border-t border-stone-200 bg-white">
      <div class="relative">
        <textarea 
          v-model="inputValue"
          @keydown.enter.prevent="sendMessage"
          :placeholder="isRAGMode ? '向知识库提问...' : '输入指令或问题...'"
          class="w-full resize-none rounded-lg border border-stone-200 bg-stone-50 p-3 pr-10 text-sm text-stone-800 focus:border-[#D06847] focus:outline-none focus:ring-1 focus:ring-[#D06847] placeholder-stone-400 transition-all"
          rows="3"
        ></textarea>
        <button 
          @click="sendMessage"
          class="absolute bottom-3 right-3 p-1.5 rounded-md bg-[#D06847] text-white hover:bg-[#B55739] transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!inputValue.trim()"
        >
          <Send class="w-3 h-3" />
        </button>
      </div>
      <div class="mt-3 flex justify-between items-center px-1">
        <!-- 知识库 Toggle Switch -->
        <div class="flex items-center space-x-2">
          <button 
            @click="isRAGMode = !isRAGMode"
            class="relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none"
            :class="isRAGMode ? 'bg-[#D06847]' : 'bg-stone-300'"
            title="点击切换知识库模式"
          >
            <span class="sr-only">使用知识库</span>
            <span
              aria-hidden="true"
              class="pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
              :class="isRAGMode ? 'translate-x-4' : 'translate-x-0'"
            ></span>
          </button>
          <span 
            class="text-xs font-medium cursor-pointer select-none"
            :class="isRAGMode ? 'text-[#D06847]' : 'text-stone-500'"
            @click="isRAGMode = !isRAGMode"
          >
            知识库
          </span>
        </div>
        
        <span class="text-xs text-stone-400">
          {{ isRAGMode ? '已启用本地检索' : '通用模型' }}
        </span>
      </div>
    </div>
  </div>
</template>
