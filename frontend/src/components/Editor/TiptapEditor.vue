<script setup lang="ts">
import { onBeforeUnmount, watch } from 'vue';
import { useEditor, EditorContent } from '@tiptap/vue-3';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import Typography from '@tiptap/extension-typography';
import { Markdown } from 'tiptap-markdown';
import { 
  Bold, Italic, Heading1, Heading2, Heading3, 
  List, ListOrdered, Code, Quote, Undo, Redo 
} from 'lucide-vue-next';
import { useEditorStore } from '../../stores/editorStore';

const store = useEditorStore();

const getInitialContent = () => {
    if (!store.currentDocument) return '';
    return store.currentDocument.content || '';
};

const editor = useEditor({
  content: getInitialContent(),
  extensions: [
    StarterKit,
    Typography,
    Placeholder.configure({
      placeholder: 'Type / for commands...',
    }),
    Markdown.configure({
      transformPastedText: true,
      transformCopiedText: true,
    }),
  ],
  editorProps: {
    attributes: {
      class: 'prose prose-stone max-w-none focus:outline-none py-12 px-12 text-[#171717] prose-headings:font-serif prose-headings:text-[#171717] prose-headings:font-bold prose-headings:tracking-tight prose-p:text-[#171717] prose-p:font-sans prose-p:font-normal prose-p:leading-relaxed prose-strong:text-[#171717] prose-strong:font-bold prose-code:text-[#DA7756] prose-code:bg-stone-50 prose-code:rounded-md prose-code:px-1.5 prose-code:py-0.5 prose-code:font-mono prose-code:font-medium prose-blockquote:border-l-4 prose-blockquote:border-[#DA7756] prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-stone-500',
    },
  },
  onUpdate: ({ editor }) => {
    const markdown = (editor.storage as any).markdown.getMarkdown();
    store.updateContent(markdown);
    const text = editor.getText();
    store.wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
  },
  onSelectionUpdate: ({ editor }) => {
    const { from } = editor.state.selection;
    // Simple line/col estimation
    store.cursorPosition = { line: 1, col: from };
  }
});

watch(() => store.currentDocument?.id, (newId) => {
    if (newId && editor.value) {
        const newContent = getInitialContent();
        if ((editor.value.storage as any).markdown.getMarkdown() !== newContent) {
            editor.value.commands.setContent(newContent);
        }
    }
});

const downloadMarkdown = () => {
  if (!store.currentDocument || !editor.value) return;
  const markdown = (editor.value.storage as any).markdown.getMarkdown();
  const docTitle = store.currentDocument.title || 'untitled';
  const blob = new Blob([markdown], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${docTitle}.md`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

defineExpose({ downloadMarkdown });


const tools = [
  { icon: Bold, title: '加粗', action: () => editor.value?.chain().focus().toggleBold().run(), isActive: () => editor.value?.isActive('bold') },
  { icon: Italic, title: '斜体', action: () => editor.value?.chain().focus().toggleItalic().run(), isActive: () => editor.value?.isActive('italic') },
  { icon: Heading1, title: '标题 1', action: () => editor.value?.chain().focus().toggleHeading({ level: 1 }).run(), isActive: () => editor.value?.isActive('heading', { level: 1 }) },
  { icon: Heading2, title: '标题 2', action: () => editor.value?.chain().focus().toggleHeading({ level: 2 }).run(), isActive: () => editor.value?.isActive('heading', { level: 2 }) },
  { icon: Heading3, title: '标题 3', action: () => editor.value?.chain().focus().toggleHeading({ level: 3 }).run(), isActive: () => editor.value?.isActive('heading', { level: 3 }) },
  { icon: List, title: '无序列表', action: () => editor.value?.chain().focus().toggleBulletList().run(), isActive: () => editor.value?.isActive('bulletList') },
  { icon: ListOrdered, title: '有序列表', action: () => editor.value?.chain().focus().toggleOrderedList().run(), isActive: () => editor.value?.isActive('orderedList') },
  { icon: Code, title: '代码块', action: () => editor.value?.chain().focus().toggleCodeBlock().run(), isActive: () => editor.value?.isActive('codeBlock') },
  { icon: Quote, title: '引用', action: () => editor.value?.chain().focus().toggleBlockquote().run(), isActive: () => editor.value?.isActive('blockquote') },
  { icon: Undo, title: '撤销', action: () => editor.value?.chain().focus().undo().run(), isActive: () => false },
  { icon: Redo, title: '重做', action: () => editor.value?.chain().focus().redo().run(), isActive: () => false },
];

onBeforeUnmount(() => {
  editor.value?.destroy();
});
</script>

<template>
  <div class="flex flex-col w-full h-full bg-[#F5F5F5] relative">
    <!-- Fixed Toolbar -->
    <div v-if="editor" class="flex items-center justify-center space-x-1 px-4 py-2 border-b border-stone-200 bg-[#F5F5F5] sticky top-0 z-20 h-12">
      <div class="flex items-center space-x-0.5 p-1">
        <button
          v-for="(tool, index) in tools"
          :key="index"
          @click="tool.action"
          class="p-2 rounded-md transition-all duration-200 text-stone-500 hover:bg-stone-200 hover:text-stone-900 focus:outline-none font-medium"
          :class="{ 'bg-stone-200 text-[#D06847] shadow-sm': tool.isActive() }"
          :title="tool.title"
        >
          <component :is="tool.icon" class="w-4 h-4" />
        </button>
      </div>
    </div>

    <!-- Editor Content Wrapper -->
    <div class="flex-1 overflow-y-auto flex justify-center p-8 cursor-default" @click.self="editor?.commands.focus()">
      <!-- Paper Sheet -->
      <div class="w-full max-w-3xl bg-white min-h-[calc(100vh-10rem)] shadow-sm border border-stone-100 cursor-text">
         <editor-content :editor="editor" class="h-full" />
      </div>
    </div>
  </div>
</template>

<style>
/* Custom Scrollbar for editor */
.ProseMirror::-webkit-scrollbar {
  width: 8px;
}
.ProseMirror::-webkit-scrollbar-track {
  background: transparent;
}
.ProseMirror::-webkit-scrollbar-thumb {
  background-color: #e5e5e4;
  border-radius: 20px;
}
.ProseMirror::-webkit-scrollbar-thumb:hover {
  background-color: #d6d3d1;
}
</style>
