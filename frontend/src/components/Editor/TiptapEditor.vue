<script setup lang="ts">
import { useEditor, EditorContent } from '@tiptap/vue-3'
import { BubbleMenu } from '@tiptap/vue-3/menus'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from 'tiptap-markdown'
import Typography from '@tiptap/extension-typography'
import Image from '@tiptap/extension-image'
import { Table } from '@tiptap/extension-table'
import { TableRow } from '@tiptap/extension-table-row'
import { TableCell } from '@tiptap/extension-table-cell'
import { TableHeader } from '@tiptap/extension-table-header'
import TaskList from '@tiptap/extension-task-list'
import TaskItem from '@tiptap/extension-task-item'
import { watch, onBeforeUnmount, ref } from 'vue'
import { useEditorStore } from '../../stores/editorStore'
import { useDebounceFn } from '@vueuse/core'
import MarkdownIt from 'markdown-it'
import taskLists from 'markdown-it-task-lists'
import { 
  Bold, Italic, Strikethrough, Code, Heading1, Heading2, Heading3, 
  List, ListOrdered, Quote, Minus, Table as TableIcon, CheckSquare, 
  Trash2, Merge, Split, ArrowDown, ArrowRight, X
} from 'lucide-vue-next'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])
const store = useEditorStore()
const isComposing = ref(false)
const md = new MarkdownIt({ html: true, breaks: true })
md.use(taskLists)

const debouncedSave = useDebounceFn(() => {
  // Save logic handled by store or parent usually
}, 1000)

// 统计字数与光标位置
const updateStats = (editor: any) => {
  const text = editor.state.doc.textContent
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0
  
  const selection = editor.state.selection
  const { from } = selection
  const textBefore = editor.state.doc.textBetween(0, from, '\n')
  const lines = textBefore.split('\n')
  const line = lines.length
  const col = lines[lines.length - 1].length + 1
  
  store.updateStats(wordCount, line, col)
}

const editor = useEditor({
  content: props.modelValue, // 直接传入，让 extension 处理 hydration
  extensions: [
    StarterKit.configure({ 
      // 修复 1: 开启所有标题级别支持 (H1-H6)
      heading: { levels: [1, 2, 3, 4, 5, 6] }, 
    }),
    Markdown.configure({ 
      html: true, 
      transformPastedText: true, 
      transformCopiedText: true, 
    }), 
    Typography, 
    Image, 
    Table.configure({ resizable: true }), 
    TableRow, 
    TableHeader, 
    TableCell, 
    TaskList, 
    TaskItem.configure({ nested: true }), 
  ], 
  editorProps: { 
    attributes: { 
      class: 'prose prose-stone max-w-none focus:outline-none min-h-[calc(100vh-12rem)] px-12 py-10 bg-white shadow-sm mx-auto', 
    }, 
    // 修复 2: 强力 Markdown 粘贴拦截 
    handlePaste: (_view, event) => { 
      const text = event.clipboardData?.getData('text/plain') 
      const html = event.clipboardData?.getData('text/html') 
      
      // 如果剪贴板里有 HTML，通常优先用 HTML（保留样式）。 
      // 但如果用户明显是在粘贴一段 Markdown 源码（比如从代码编辑器复制过来的）， 
      // 我们需要探测 Markdown 特征。 
      if (text) { 
        // 特征检测：行首的 #, >, -, *, 1., ``` 
        const isMarkdownLike = /^(\s*#{1,6}\s|\s*>|\s*[-*]\s|\s*\d+\.\s|`{3})/.test(text) 
        
        // 如果像 Markdown，且 HTML 内容很少（说明不是从富文本编辑器复制的），强制解析 
        if (isMarkdownLike && (!html || html.length < text.length * 1.5)) { 
          const parsedHtml = md.render(text) 
          editor.value?.commands.insertContent(parsedHtml) 
          return true // 阻止默认行为 
        } 
      } 
      return false 
    } 
  }, 
  onUpdate: ({ editor }) => { 
    if (isComposing.value) return 
    const markdown = (editor.storage as any).markdown.getMarkdown() 
    emit('update:modelValue', markdown) 
    store.updateContent(markdown) 
    debouncedSave() 
    updateStats(editor) 
  }, 
  onSelectionUpdate: ({ editor }) => {
    updateStats(editor)
  },
}) 

// 双向绑定监听 
watch(() => props.modelValue, (newValue) => { 
  if (editor.value) { 
    const currentMarkdown = (editor.value.storage as any).markdown.getMarkdown() 
    if (newValue !== currentMarkdown) { 
      editor.value.commands.setContent(newValue) 
    } 
  } 
}) 

const onCompositionStart = () => { isComposing.value = true } 
const onCompositionEnd = () => { isComposing.value = false } 

// --- Actions --- 
const toggleBold = () => editor.value?.chain().focus().toggleBold().run() 
const toggleItalic = () => editor.value?.chain().focus().toggleItalic().run() 
const toggleStrike = () => editor.value?.chain().focus().toggleStrike().run() 
const toggleCode = () => editor.value?.chain().focus().toggleCode().run() 
const toggleH1 = () => editor.value?.chain().focus().toggleHeading({ level: 1 }).run() 
const toggleH2 = () => editor.value?.chain().focus().toggleHeading({ level: 2 }).run() 
const toggleH3 = () => editor.value?.chain().focus().toggleHeading({ level: 3 }).run() 
const toggleBulletList = () => editor.value?.chain().focus().toggleBulletList().run() 
const toggleOrderedList = () => editor.value?.chain().focus().toggleOrderedList().run() 
const toggleTaskList = () => editor.value?.chain().focus().toggleTaskList().run() 
const toggleBlockquote = () => editor.value?.chain().focus().toggleBlockquote().run() 
const setHorizontalRule = () => editor.value?.chain().focus().setHorizontalRule().run() 
// 修改插入表格逻辑：保留默认 3x3 快捷键 
const insertTable = () => editor.value?.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run() 

// Table Operations 
const addColumnAfter = () => editor.value?.chain().focus().addColumnAfter().run() 
const deleteColumn = () => editor.value?.chain().focus().deleteColumn().run() 
const addRowAfter = () => editor.value?.chain().focus().addRowAfter().run() 
const deleteRow = () => editor.value?.chain().focus().deleteRow().run()
const deleteTable = () => editor.value?.chain().focus().deleteTable().run()
const mergeCells = () => editor.value?.chain().focus().mergeCells().run()
const splitCell = () => editor.value?.chain().focus().splitCell().run()

onBeforeUnmount(() => editor.value?.destroy())
</script>

<template>
  <div class="flex flex-col h-full w-full bg-[#F9F9F9] relative overflow-hidden">
    <!-- Toolbar -->
    <div v-if="editor" class="flex items-center justify-center p-2 border-b border-stone-200 bg-white z-10 shadow-sm flex-shrink-0">
      <div class="flex items-center gap-1 overflow-x-auto max-w-5xl w-full px-2">
        <button @click="toggleBold" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('bold') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="加粗">
          <Bold class="w-4 h-4" />
        </button>
        <button @click="toggleItalic" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('italic') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="斜体">
          <Italic class="w-4 h-4" />
        </button>
        <button @click="toggleStrike" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('strike') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="删除线">
          <Strikethrough class="w-4 h-4" />
        </button>
        <button @click="toggleCode" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('code') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="代码">
          <Code class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="toggleH1" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 1 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 1">
          <Heading1 class="w-4 h-4" />
        </button>
        <button @click="toggleH2" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 2 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 2">
          <Heading2 class="w-4 h-4" />
        </button>
        <button @click="toggleH3" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('heading', { level: 3 }) }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="标题 3">
          <Heading3 class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="toggleBulletList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('bulletList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="无序列表">
          <List class="w-4 h-4" />
        </button>
        <button @click="toggleOrderedList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('orderedList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="有序列表">
          <ListOrdered class="w-4 h-4" />
        </button>
        <button @click="toggleTaskList" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('taskList') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="任务列表">
          <CheckSquare class="w-4 h-4" />
        </button>
        <button @click="toggleBlockquote" :class="{ 'bg-stone-100 text-stone-900': editor.isActive('blockquote') }" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="引用">
          <Quote class="w-4 h-4" />
        </button>
        <div class="w-px h-4 bg-stone-200 mx-1"></div>
        <button @click="insertTable" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="插入表格 (3x3)">
          <TableIcon class="w-4 h-4" />
        </button>
        <button @click="setHorizontalRule" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="水平分割线">
          <Minus class="w-4 h-4" />
        </button>

        <!-- Dynamic Table Toolbar -->
        <div v-if="editor.isActive('table')" class="flex items-center gap-1 ml-2 pl-2 border-l border-stone-200 animate-in fade-in slide-in-from-left-2 duration-200">
          <div class="flex items-center gap-0.5">
            <button @click="addRowAfter" class="p-1.5 rounded hover:bg-stone-100 text-primary" title="下方插入行">
              <ArrowDown class="w-4 h-4" />
            </button>
            <button @click="deleteRow" class="p-1.5 rounded hover:bg-red-50 text-red-500" title="删除当前行">
              <X class="w-4 h-4" />
            </button>
          </div>
          <div class="w-px h-3 bg-stone-200 mx-0.5"></div>
          <div class="flex items-center gap-0.5">
            <button @click="addColumnAfter" class="p-1.5 rounded hover:bg-stone-100 text-primary" title="右侧插入列">
              <ArrowRight class="w-4 h-4" />
            </button>
            <button @click="deleteColumn" class="p-1.5 rounded hover:bg-red-50 text-red-500" title="删除当前列">
              <X class="w-4 h-4" />
            </button>
          </div>
          <div class="w-px h-3 bg-stone-200 mx-0.5"></div>
          <button @click="mergeCells" class="p-1.5 rounded hover:bg-stone-100 text-stone-600" title="合并单元格">
            <Merge class="w-4 h-4" />
          </button>
           <button @click="splitCell" class="p-1.5 rounded hover:bg-stone-100 text-stone-600" title="拆分单元格">
            <Split class="w-4 h-4" />
          </button>
          <div class="w-px h-3 bg-stone-200 mx-0.5"></div>
          <button @click="deleteTable" class="p-1.5 rounded hover:bg-red-50 text-red-500" title="删除整个表格">
            <Trash2 class="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>

    <div class="flex-1 overflow-y-auto custom-scrollbar p-8">
      <div class="max-w-4xl mx-auto w-full">
        <editor-content
          :editor="editor"
          class="h-full"
          @compositionstart="onCompositionStart"
          @compositionend="onCompositionEnd"
        />
      </div>
    </div>

    <!-- Bubble Menu (Optional but kept for selection context) -->
    <bubble-menu
      v-if="editor && !editor.isActive('table')"
      :editor="editor"
      :tippy-options="{ duration: 100 }"
      class="bg-white border border-stone-200 shadow-xl rounded-lg overflow-hidden flex divide-x divide-stone-100"
    >
      <button @click="editor.chain().focus().toggleBold().run()" :class="{ 'bg-stone-100': editor.isActive('bold') }" class="p-2 hover:bg-stone-50 text-sm font-medium text-stone-600">加粗</button>
      <button @click="editor.chain().focus().toggleItalic().run()" :class="{ 'bg-stone-100': editor.isActive('italic') }" class="p-2 hover:bg-stone-50 text-sm font-medium text-stone-600">斜体</button>
      <button @click="editor.chain().focus().toggleCodeBlock().run()" :class="{ 'bg-stone-100': editor.isActive('codeBlock') }" class="p-2 hover:bg-stone-50 text-sm font-medium text-stone-600">代码块</button>
    </bubble-menu>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-thumb { background-color: #e5e5e5; border-radius: 3px; }
.rotate-90 { transform: rotate(90deg); }
</style>
