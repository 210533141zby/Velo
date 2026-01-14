<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import { useEditorStore } from '../../stores/editorStore';
import { X, Send, Bot, User as UserIcon, Database } from 'lucide-vue-next';

const store = useEditorStore();
const inputValue = ref('');
const isLoading = ref(false);
const useRAG = ref(false); // Default to AI (False), Toggle ON for RAG
const messages = ref([
  { id: 1, role: 'ai', content: '你好！我是 Wiki AI 助手。今天有什么可以帮你的吗？' }
]);

const sendMessage = async () => {
  if (!inputValue.value.trim() || isLoading.value) return;
  
  const userText = inputValue.value;
  inputValue.value = '';

  // Add User Message
  messages.value.push({
    id: Date.now(),
    role: 'user',
    content: userText
  });
  
  isLoading.value = true;

  try {
    // Call Backend API
    const response = await axios.post('http://localhost:8000/api/v1/agent/chat', {
      messages: [{ role: 'user', content: userText }],
      use_rag: useRAG.value
    });

    const aiResponse = response.data.response;

    messages.value.push({
      id: Date.now() + 1,
      role: 'ai',
      content: aiResponse
    });
  } catch (error) {
    console.error('Chat error:', error);
    messages.value.push({
      id: Date.now() + 1,
      role: 'ai',
      content: '抱歉，处理您的请求时出现错误。'
    });
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="h-full flex flex-col bg-zinc-50 border-l border-zinc-200/60 shadow-xl">
    <!-- Header -->
    <div class="h-12 flex items-center justify-between px-4 border-b border-zinc-200/60 bg-white/50 backdrop-blur">
      <div class="flex items-center space-x-2">
        <Bot class="w-4 h-4 text-primary" />
        <span class="font-semibold text-zinc-800 text-sm">AI 助手</span>
      </div>
      
      <div class="flex items-center space-x-2">
        <button 
          @click="store.toggleCopilot"
          class="text-zinc-400 hover:text-zinc-600 transition-colors"
        >
          <X class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Messages -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      <div 
        v-for="msg in messages" 
        :key="msg.id"
        class="flex items-start space-x-3"
        :class="msg.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''"
      >
        <div 
          class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
          :class="msg.role === 'ai' ? 'bg-orange-50 text-primary' : 'bg-zinc-200 text-zinc-600'"
        >
          <Bot v-if="msg.role === 'ai'" class="w-4 h-4" />
          <UserIcon v-else class="w-4 h-4" />
        </div>
        
        <div 
          class="max-w-[80%] rounded-lg p-3 text-sm leading-relaxed"
          :class="msg.role === 'ai' ? 'bg-white border border-zinc-200/60 text-zinc-700 shadow-sm' : 'bg-primary text-white shadow-md'"
        >
          {{ msg.content }}
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="p-3 border-t border-zinc-200/60 bg-white">
      <div class="relative flex flex-col gap-2">
        <textarea 
          v-model="inputValue"
          @keydown.enter.prevent="sendMessage"
          placeholder="输入您的问题..."
          class="w-full resize-none rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-sm text-zinc-800 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
          rows="3"
        ></textarea>
        
        <div class="flex justify-between items-center mt-1">
           <!-- Knowledge Base Toggle -->
           <div class="flex items-center space-x-2 cursor-pointer group select-none" @click="useRAG = !useRAG">
              <div 
                class="relative inline-flex h-5 w-9 items-center rounded-full transition-colors duration-200 ease-in-out border border-transparent ring-1 ring-inset"
                :class="useRAG ? 'bg-primary ring-primary' : 'bg-zinc-200 ring-zinc-200'"
              >
                <span
                  class="inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"
                  :class="useRAG ? 'translate-x-4' : 'translate-x-0.5'"
                />
              </div>
              <span class="text-xs font-medium text-zinc-500 group-hover:text-zinc-700 transition-colors">知识库</span>
           </div>

           <button 
              @click="sendMessage"
              class="flex items-center justify-center p-2 rounded-md bg-primary text-white hover:opacity-90 transition-opacity shadow-sm disabled:opacity-50"
              :disabled="isLoading || !inputValue.trim()"
            >
              <Send class="w-3.5 h-3.5" />
           </button>
        </div>
      </div>
    </div>
  </div>
</template>
