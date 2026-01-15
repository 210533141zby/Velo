<script setup lang="ts">
import { ref } from 'vue';
import { useEditorStore } from '../../stores/editorStore';
import { X, Send, Bot, User as UserIcon } from 'lucide-vue-next';

const store = useEditorStore();
const inputValue = ref('');
const messages = ref([
  { id: 1, role: 'ai', content: 'Hello! I am Velo AI. How can I help you edit this document today?' }
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
  inputValue.value = '';

  // Simulate AI Response
  setTimeout(() => {
    messages.value.push({
      id: Date.now() + 1,
      role: 'ai',
      content: `I received your message: "${userText}". This is a mock response.`
    });
  }, 1000);
};
</script>

<template>
  <div class="h-full flex flex-col bg-zinc-50 border-l border-zinc-200/60 shadow-xl">
    <!-- Header -->
    <div class="h-12 flex items-center justify-between px-4 border-b border-zinc-200/60 bg-white/50 backdrop-blur">
      <div class="flex items-center space-x-2">
        <Bot class="w-4 h-4 text-indigo-600" />
        <span class="font-semibold text-zinc-800 text-sm">AI Assistant</span>
      </div>
      <button 
        @click="store.toggleCopilot"
        class="text-zinc-400 hover:text-zinc-600 transition-colors"
      >
        <X class="w-4 h-4" />
      </button>
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
          :class="msg.role === 'ai' ? 'bg-indigo-100 text-indigo-600' : 'bg-zinc-200 text-zinc-600'"
        >
          <Bot v-if="msg.role === 'ai'" class="w-4 h-4" />
          <UserIcon v-else class="w-4 h-4" />
        </div>
        
        <div 
          class="max-w-[80%] rounded-lg p-3 text-sm leading-relaxed"
          :class="msg.role === 'ai' ? 'bg-white border border-zinc-200/60 text-zinc-700 shadow-sm' : 'bg-indigo-600 text-white shadow-md'"
        >
          {{ msg.content }}
        </div>
      </div>
    </div>

    <!-- Input -->
    <div class="p-4 border-t border-zinc-200/60 bg-white">
      <div class="relative">
        <textarea 
          v-model="inputValue"
          @keydown.enter.prevent="sendMessage"
          placeholder="Ask AI anything..."
          class="w-full resize-none rounded-lg border border-zinc-200 bg-zinc-50 p-3 pr-10 text-sm text-zinc-800 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          rows="3"
        ></textarea>
        <button 
          @click="sendMessage"
          class="absolute bottom-3 right-3 p-1.5 rounded-md bg-indigo-600 text-white hover:bg-indigo-700 transition-colors shadow-sm"
        >
          <Send class="w-3 h-3" />
        </button>
      </div>
    </div>
  </div>
</template>
