<script setup lang="ts">
import { useEditorStore } from '../../stores/editorStore';

const store = useEditorStore();
</script>

<template>
  <div class="h-8 bg-surface border-t border-border-line text-text-sub flex items-center justify-between px-4 text-[11px] select-none font-sans">
    <!-- Left -->
    <div class="flex items-center space-x-4">
       <div class="hover:bg-neutral-50 px-2 py-1 rounded-md cursor-pointer transition-colors font-medium border border-transparent hover:border-neutral-200">
        Words: {{ store.wordCount }}
      </div>
       <div class="hover:bg-neutral-50 px-2 py-1 rounded-md cursor-pointer transition-colors font-medium border border-transparent hover:border-neutral-200">
        Ln {{ store.cursorPosition.line }}, Col {{ store.cursorPosition.col }}
      </div>
       <div class="hover:bg-neutral-50 px-2 py-1 rounded-md cursor-pointer transition-colors font-medium border border-transparent hover:border-neutral-200">
        UTF-8
      </div>
    </div>

    <!-- Right -->
    <div class="flex items-center space-x-4">
      <div class="hidden sm:block hover:bg-neutral-50 px-2 py-1 rounded-md cursor-pointer transition-colors font-medium text-text-sub">
        Markdown
      </div>
      <div class="hover:bg-neutral-50 px-2 py-1 rounded-md cursor-pointer transition-colors flex items-center space-x-2 border border-border-line bg-surface shadow-sm">
        <span class="w-1.5 h-1.5 rounded-full bg-emerald-500 ring-2 ring-emerald-100"></span>
        <span class="font-bold text-text-main">Velo AI Ready</span>
      </div>
    </div>
  </div>
</template>
