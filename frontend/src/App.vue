<script setup lang="ts">
import { useEditorStore } from './stores/editorStore';
import FileTree from './components/Sidebar/FileTree.vue';
import TiptapEditor from './components/Editor/TiptapEditor.vue';
import ChatPanel from './components/Copilot/ChatPanel.vue';
import StatusBar from './components/Layout/StatusBar.vue';
import { PanelLeft, Sparkles, Save, Share, Loader2, Check, Bot, File } from 'lucide-vue-next';

const store = useEditorStore();
</script>

<template>
  <div class="flex h-screen w-full bg-[#F5F5F5] overflow-hidden text-stone-900 font-sans">
    <!-- Sidebar -->
    <aside 
      class="flex-shrink-0 transition-all duration-300 ease-in-out border-r border-stone-200 bg-[#F5F5F5]"
      :class="store.isSidebarOpen ? 'w-[240px]' : 'w-0 border-none overflow-hidden opacity-0'"
    >
      <FileTree />
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col min-w-0 bg-[#F5F5F5] relative z-0">
      <!-- Workbench Header -->
      <div class="h-14 flex items-center justify-between border-b border-stone-200 px-6 bg-[#F5F5F5] z-20">
        <div class="flex items-center h-full space-x-4">
          <!-- Sidebar Toggle -->
          <button 
            @click="store.toggleSidebar"
            class="h-9 w-9 flex items-center justify-center rounded-lg text-stone-500 hover:text-stone-900 hover:bg-stone-50 transition-colors"
            title="Toggle Sidebar"
          >
            <PanelLeft class="w-5 h-5" />
          </button>
          
          <!-- Breadcrumbs / Title -->
          <div class="flex items-center text-sm text-stone-500 space-x-2 font-medium">
            <div class="hidden sm:flex items-center space-x-2 hover:text-stone-900 cursor-pointer transition-colors">
              <Sparkles class="w-4 h-4 text-[#D06847]" />
              <span class="font-serif font-bold text-stone-900">Velo</span>
            </div>
            <span class="text-stone-300 hidden sm:inline-block">/</span>
            <!-- Editable Title Input -->
            <input 
              v-if="store.currentDocument"
              v-model="store.currentDocument.title"
              @input="store.updateTitle(store.currentDocument.title)"
              class="font-serif font-bold text-stone-900 text-lg bg-transparent border-b-2 border-transparent hover:border-stone-200 focus:border-[#D06847] focus:outline-none px-1 py-0.5 transition-all w-[300px] placeholder-stone-300"
              placeholder="Enter document title..."
            />
            <span v-else class="font-serif font-bold text-stone-500 text-lg px-1">No Document Selected</span>
          </div>
        </div>
        
        <!-- Right Actions -->
        <div class="flex items-center space-x-3">
          <!-- Save Button -->
           <button 
            @click="store.saveCurrentDocument"
            class="flex items-center justify-center w-8 h-8 rounded-lg transition-all hover:bg-stone-100"
            :class="{
              'cursor-wait': store.saveStatus === 'saving',
            }"
            :title="store.saveStatus === 'saving' ? 'Saving...' : (store.saveStatus === 'saved' ? 'Saved' : 'Save')"
          >
            <Loader2 v-if="store.saveStatus === 'saving'" class="w-4 h-4 animate-spin text-stone-400" />
            <Check v-else-if="store.saveStatus === 'saved'" class="w-4 h-4 text-green-600" />
            <Save v-else class="w-4 h-4 text-stone-400" />
          </button>

          <!-- Export/Share -->
           <button 
            class="p-2 rounded-lg text-stone-500 hover:text-stone-900 hover:bg-stone-50 transition-colors border border-transparent hover:border-stone-200"
            title="Export PDF"
          >
            <Share class="w-5 h-5" />
          </button>

          <div class="w-px h-5 bg-stone-200 mx-1"></div>

          <!-- AI Toggle -->
          <button 
            @click="store.toggleCopilot"
            class="flex items-center space-x-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors border"
            :class="store.isCopilotOpen ? 'bg-[#D06847]/10 text-[#D06847] border-[#D06847]' : 'text-stone-500 border-transparent hover:bg-stone-50 hover:text-stone-900'"
          >
            <Bot class="w-4 h-4" />
            <span>AI</span>
          </button>
        </div>
      </div>

      <!-- Editor Area -->
      <div class="flex-1 relative overflow-hidden bg-white">
        <TiptapEditor 
          v-if="store.currentDocument"
          :modelValue="store.currentDocument.content"
          @update:modelValue="(val) => store.updateContent(val)"
        />
        <div v-else class="flex flex-col items-center justify-center h-full text-stone-400 bg-stone-50/50">
          <File class="w-12 h-12 mb-4 opacity-20" />
          <p class="text-lg font-medium opacity-60">No document selected</p>
          <p class="text-sm opacity-40 mt-1">Select a file from the sidebar to start editing</p>
        </div>
      </div>

      <!-- Status Bar -->
      <StatusBar />
    </main>

    <!-- Copilot Panel -->
    <aside 
      class="flex-shrink-0 transition-all duration-300 ease-in-out border-l border-border-line bg-surface"
      :class="store.isCopilotOpen ? 'w-[350px]' : 'w-0 border-none overflow-hidden opacity-0'"
    >
      <ChatPanel />
    </aside>
  </div>
</template>
