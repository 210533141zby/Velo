<script setup lang="ts">
import { formatDate } from '@/utils/date';
import type { Folder as FolderType, Doc } from '@/api';
import { 
  Folder, 
  FileText, 
  Trash2, 
  FolderInput, 
  FolderPlus
} from 'lucide-vue-next';

const props = defineProps<{
  folders: FolderType[];
  documents: Doc[];
  currentView: string;
  breadcrumbs: { id: number | null; title: string }[];
}>();

const emit = defineEmits<{
  (e: 'update:currentView', view: string): void;
  (e: 'openFolder', folder: FolderType): void;
  (e: 'createFolder'): void;
  (e: 'createNewDoc'): void;
  (e: 'selectDoc', doc: Doc): void;
  (e: 'navigateBreadcrumb', index: number): void;
  (e: 'deleteDoc', doc: Doc): void;
  (e: 'moveDoc', doc: Doc): void;
}>();

const handleOpenFolder = (folder: FolderType) => {
  emit('openFolder', folder);
};
</script>

<template>
  <div class="flex-1 overflow-y-auto px-8 py-8 bg-white">
    <!-- Empty State -->
    <div v-if="documents.length === 0 && folders.length === 0" class="h-full flex flex-col items-center justify-center text-zinc-400">
      <div class="w-16 h-16 bg-zinc-50 rounded-2xl flex items-center justify-center mb-4">
        <FolderPlus class="w-8 h-8 text-zinc-300" />
      </div>
      <p class="text-sm font-medium text-zinc-500 mb-6">This folder is empty</p>
      <div class="flex gap-3">
           <button @click="emit('createFolder')" class="px-4 py-2 bg-white border border-zinc-200 rounded-lg text-sm font-medium text-zinc-700 hover:border-zinc-300 hover:shadow-sm transition-all">
             New Folder
           </button>
           <button @click="emit('createNewDoc')" class="px-4 py-2 bg-zinc-900 text-white rounded-lg text-sm font-medium hover:bg-zinc-800 shadow-sm transition-all">
             New Document
           </button>
      </div>
    </div>

    <div v-else class="max-w-[1200px] mx-auto">
      <!-- Section: Folders -->
      <div v-if="folders.length > 0" class="mb-10">
        <h3 class="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4 px-1">Folders</h3>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
          <div 
            v-for="folder in folders" 
            :key="folder.id" 
            @click="handleOpenFolder(folder)" 
            class="group relative flex flex-col p-4 bg-zinc-50/50 border border-zinc-100 rounded-xl hover:bg-white hover:border-zinc-200 hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300 cursor-pointer"
          >
            <div class="mb-3">
              <Folder class="w-8 h-8 text-blue-300 fill-blue-50 group-hover:text-blue-500 transition-colors" />
            </div>
            <span class="text-sm font-semibold text-zinc-700 group-hover:text-zinc-900 truncate">{{ folder.title }}</span>
            <span class="text-xs text-zinc-400 mt-1">Folder</span>
          </div>
        </div>
      </div>

      <!-- Section: Documents -->
      <div v-if="documents.length > 0">
        <h3 class="text-xs font-bold text-zinc-400 uppercase tracking-widest mb-4 px-1">Documents</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
          <div 
            v-for="doc in documents" 
            :key="doc.id" 
            @click="emit('selectDoc', doc)" 
            class="group relative bg-white p-6 rounded-2xl border border-zinc-100 hover:border-zinc-200 hover:shadow-xl hover:-translate-y-1 transition-all duration-300 cursor-pointer flex flex-col h-[240px]"
          >
            <!-- Card Header -->
            <div class="flex justify-between items-start mb-4">
              <div class="w-10 h-10 rounded-xl bg-zinc-50 flex items-center justify-center text-zinc-400 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors duration-300">
                <FileText class="w-5 h-5" />
              </div>
              
              <!-- Actions (Hidden by default, show on hover) -->
              <div class="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 translate-x-2 group-hover:translate-x-0">
                  <button @click.stop="emit('moveDoc', doc)" class="p-1.5 rounded-lg hover:bg-zinc-100 text-zinc-400 hover:text-zinc-700 transition-colors" title="Move">
                      <FolderInput class="w-4 h-4" />
                  </button>
                  <button @click.stop="emit('deleteDoc', doc)" class="p-1.5 rounded-lg hover:bg-red-50 text-zinc-400 hover:text-red-500 transition-colors" title="Delete">
                      <Trash2 class="w-4 h-4" />
                  </button>
              </div>
            </div>
            
            <!-- Content -->
            <h3 class="font-bold text-zinc-800 text-lg mb-2 line-clamp-1 group-hover:text-blue-600 transition-colors">{{ doc.title || 'Untitled' }}</h3>
            <p class="text-xs text-zinc-400 mb-4 font-medium flex items-center gap-1">
              <span>Edited {{ formatDate(doc.updated_at || doc.created_at) }}</span>
            </p>
            <p class="text-sm text-zinc-500 leading-relaxed line-clamp-3 flex-1 overflow-hidden">{{ doc.content || 'No content' }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
