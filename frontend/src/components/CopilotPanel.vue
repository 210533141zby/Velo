<script setup lang="ts">
import { ref, nextTick } from 'vue';
import { chatWithAgent } from '@/api';
import { 
  Send, 
  Sparkles, 
  Bot, 
  User, 
  PanelRightClose, 
  PanelRightOpen,
  Search
} from 'lucide-vue-next';

const props = defineProps<{
  collapsed: boolean;
}>();

const emit = defineEmits<{
  (e: 'toggle'): void;
}>();

const userQuery = ref('');
const isTyping = ref(false);
const useRAG = ref(false);
const chatContainer = ref<HTMLElement | null>(null);

const chatMessages = ref<{ role: string; content: string }[]>([
  { role: 'assistant', content: '你好！我是您的 AI 写作助手。我可以帮您优化文案、搜索资料或提供灵感。' }
]);

const scrollToBottom = () => {
  if (chatContainer.value) {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight;
  }
};

const sendMessage = async () => {
  if (!userQuery.value.trim() || isTyping.value) return;
  
  const query = userQuery.value;
  chatMessages.value.push({ role: 'user', content: query });
  userQuery.value = '';
  isTyping.value = true;
  
  await nextTick();
  scrollToBottom();

  try {
    const data = await chatWithAgent(query, useRAG.value);
    chatMessages.value.push({ role: 'assistant', content: data.response });
  } catch (error) {
    chatMessages.value.push({ role: 'assistant', content: '抱歉，我遇到了一些问题，请稍后再试。' });
  } finally {
    isTyping.value = false;
    await nextTick();
    scrollToBottom();
  }
};
</script>

<template>
  <aside 
    class="flex-shrink-0 bg-white border-l border-zinc-200 transition-all duration-300 ease-in-out flex flex-col h-full relative"
    :class="collapsed ? 'w-[60px]' : 'w-[320px]'"
  >
    <!-- Toggle Button (Absolute or in header) -->
    <div class="h-[60px] flex items-center justify-between px-4 border-b border-zinc-100 flex-shrink-0">
      <div v-if="!collapsed" class="flex items-center gap-2 text-zinc-800 font-semibold">
        <Sparkles class="w-4 h-4 text-blue-600" />
        <span>Copilot</span>
      </div>
      <button 
        @click="emit('toggle')" 
        class="p-2 hover:bg-zinc-100 rounded-md text-zinc-400 hover:text-zinc-600 transition-colors"
        :class="{ 'mx-auto': collapsed }"
      >
        <PanelRightOpen v-if="collapsed" class="w-5 h-5" />
        <PanelRightClose v-else class="w-5 h-5" />
      </button>
    </div>

    <!-- Collapsed State Content -->
    <div v-if="collapsed" class="flex-1 flex flex-col items-center pt-4 gap-4">
      <button class="p-3 bg-blue-50 text-blue-600 rounded-xl hover:bg-blue-100 transition-colors" title="新对话">
        <Bot class="w-5 h-5" />
      </button>
    </div>

    <!-- Expanded State Content -->
    <div v-else class="flex-1 flex flex-col min-h-0">
      <!-- Chat History -->
      <div class="flex-1 overflow-y-auto p-4 space-y-6" ref="chatContainer">
        <div v-for="(msg, index) in chatMessages" :key="index" class="flex gap-3">
          <!-- Avatar -->
          <div 
            class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 text-xs shadow-sm border border-zinc-100"
            :class="msg.role === 'user' ? 'bg-zinc-900 text-white' : 'bg-white text-blue-600'"
          >
            <User v-if="msg.role === 'user'" class="w-4 h-4" />
            <Bot v-else class="w-4 h-4" />
          </div>
          
          <!-- Message Bubble -->
          <div class="flex-1 space-y-1">
            <p class="text-xs font-bold text-zinc-400">{{ msg.role === 'user' ? 'You' : 'AI Copilot' }}</p>
            <div 
              class="text-sm leading-relaxed"
              :class="msg.role === 'user' ? 'text-zinc-800' : 'text-zinc-600'"
            >
              {{ msg.content }}
            </div>
          </div>
        </div>

        <!-- Typing Indicator -->
        <div v-if="isTyping" class="flex gap-3">
          <div class="w-8 h-8 rounded-full bg-white text-blue-600 flex items-center justify-center flex-shrink-0 border border-zinc-100">
            <Bot class="w-4 h-4" />
          </div>
          <div class="flex items-center gap-1 h-8">
            <span class="w-1.5 h-1.5 bg-zinc-300 rounded-full animate-bounce"></span>
            <span class="w-1.5 h-1.5 bg-zinc-300 rounded-full animate-bounce delay-75"></span>
            <span class="w-1.5 h-1.5 bg-zinc-300 rounded-full animate-bounce delay-150"></span>
          </div>
        </div>
      </div>

      <!-- Input Area -->
      <div class="p-4 border-t border-zinc-100 bg-zinc-50/30">
        <!-- RAG Toggle -->
        <div class="flex items-center gap-2 mb-3">
           <label class="flex items-center gap-2 cursor-pointer group">
              <div class="relative flex items-center">
                 <input type="checkbox" v-model="useRAG" class="sr-only peer">
                 <div class="w-8 h-4 bg-zinc-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-blue-600"></div>
              </div>
              <span class="text-xs font-medium text-zinc-500 group-hover:text-zinc-700 transition-colors flex items-center gap-1">
                <Search class="w-3 h-3" />
                引用知识库
              </span>
           </label>
        </div>

        <!-- Input Box -->
        <div class="relative bg-white rounded-xl border border-zinc-200 shadow-sm focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-500 transition-all">
          <textarea 
            v-model="userQuery" 
            @keydown.enter.prevent="sendMessage"
            placeholder="Ask AI for help..." 
            class="w-full pl-3 pr-10 py-3 bg-transparent border-none outline-none text-sm resize-none h-[48px] max-h-[120px] placeholder-zinc-400 text-zinc-800"
            :disabled="isTyping"
          ></textarea>
          <button 
            @click="sendMessage"
            :disabled="!userQuery.trim() || isTyping"
            class="absolute right-2 top-2 p-1.5 rounded-lg text-zinc-400 hover:text-blue-600 hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-zinc-400"
          >
            <Send class="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  </aside>
</template>
