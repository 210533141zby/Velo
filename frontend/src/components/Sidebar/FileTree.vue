<script setup lang="ts">
import { useEditorStore } from '../../stores/editorStore';
import { Plus, Settings, User, File, Trash2 } from 'lucide-vue-next';
import { onMounted, ref } from 'vue';
import ConfirmModal from '../modals/ConfirmModal.vue';

const store = useEditorStore();
const isDeleteModalVisible = ref(false);
const fileToDeleteId = ref<number | null>(null);

onMounted(() => {
  store.fetchDocuments();
});

const createNewFile = () => {
  store.createDocument();
};

const confirmDelete = (id: number, event: Event) => {
  event.stopPropagation();
  fileToDeleteId.value = id;
  isDeleteModalVisible.value = true;
};

const handleDelete = () => {
  if (fileToDeleteId.value !== null) {
    store.deleteDocument(fileToDeleteId.value);
    isDeleteModalVisible.value = false;
    fileToDeleteId.value = null;
  }
};
</script>

<template>
  <div class="h-full flex flex-col bg-[#F2F3F5] border-r border-stone-200 font-['Nunito']">
    <!-- Header -->
    <div class="h-14 flex items-center justify-between px-5 border-b border-stone-200 bg-[#F2F3F5]">
      <div class="flex items-center space-x-3">
        <div class="w-6 h-6 bg-[#D06847] rounded-lg flex items-center justify-center shadow-sm">
          <span class="text-white text-xs font-serif font-bold italic">V</span>
        </div>
        <span class="text-stone-900 font-serif font-bold tracking-tight text-lg">Velo</span>
      </div>
      <button 
        @click="createNewFile"
        class="p-1.5 hover:bg-stone-200 rounded-lg text-stone-500 hover:text-[#D06847] transition-all duration-200"
        title="Create New File"
      >
        <Plus class="w-5 h-5" />
      </button>
    </div>

    <!-- File List -->
    <div class="flex-1 overflow-y-auto py-4 px-3 bg-[#F2F3F5]">
      <div class="px-2 mb-3 text-xs font-bold text-stone-900 uppercase tracking-wider font-mono flex items-center justify-between">
        <span>Workspace</span>
        <span class="text-[10px] bg-stone-200 text-stone-900 px-1.5 py-0.5 rounded border border-stone-300 font-medium">Local</span>
      </div>
      <div class="space-y-1">
        <div 
          v-for="doc in store.documents" 
          :key="doc.id"
          @click="store.loadDocument(doc.id)"
          class="group flex items-center px-3 py-2.5 rounded-lg cursor-pointer text-sm transition-all duration-200 relative border border-transparent"
          :class="store.currentDocument?.id === doc.id ? 'bg-white text-[#D06847] font-bold shadow-sm' : 'text-stone-900 font-medium hover:bg-stone-200 hover:text-black'"
        >
          <!-- Indicator -->
          <div class="absolute left-0 top-2 bottom-2 w-1 bg-[#D06847] rounded-r transition-opacity"
            :class="store.currentDocument?.id === doc.id ? 'opacity-100' : 'opacity-0'"></div>

          <File 
            class="w-4 h-4 mr-3 transition-colors flex-shrink-0" 
            :class="store.currentDocument?.id === doc.id ? 'text-[#D06847]' : 'text-stone-500 group-hover:text-stone-700'" 
          />
          <div class="flex flex-col flex-1 min-w-0">
             <span class="truncate leading-tight">{{ doc.title || 'Untitled' }}</span>
             <span class="text-[10px] text-stone-400 truncate mt-0.5 group-hover:text-stone-500 transition-colors">
               {{ 
                 new Date(doc.updated_at ? doc.updated_at.replace(' ', 'T') + 'Z' : Date.now()).toLocaleString('zh-CN', { 
                   timeZone: 'Asia/Shanghai', 
                   hour12: false,
                   year: 'numeric',
                   month: '2-digit',
                   day: '2-digit',
                   hour: '2-digit',
                   minute: '2-digit',
                   second: '2-digit'
                 }) 
               }}
             </span>
          </div>

          <!-- Delete Button (Hover) -->
          <button 
            @click="(e) => confirmDelete(doc.id, e)"
            class="opacity-0 group-hover:opacity-100 p-1.5 text-stone-400 hover:text-red-500 hover:bg-red-50 rounded-md transition-all transform hover:scale-105"
            title="Delete File"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>

    <!-- Modals -->
      <ConfirmModal 
        v-model:visible="isDeleteModalVisible"
        message="确定要删除这个文档吗？此操作无法撤销。"
        @confirm="handleDelete"
      />

    <!-- Footer -->
    <div class="p-4 border-t border-stone-200 bg-[#F5F5F5]">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-3">
          <div class="w-8 h-8 rounded-full bg-stone-200 flex items-center justify-center overflow-hidden border border-stone-300">
             <User class="w-5 h-5 text-stone-500" />
          </div>
          <div class="flex flex-col">
            <span class="text-xs font-bold text-stone-700">User</span>
            <span class="text-[10px] text-stone-400">Pro Plan</span>
          </div>
        </div>
        <button class="p-2 hover:bg-stone-200 rounded-lg text-stone-400 hover:text-stone-600 transition-colors">
          <Settings class="w-5 h-5" />
        </button>
      </div>
    </div>
  </div>
</template>
