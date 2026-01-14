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
  Columns, Rows, Trash2, Merge, Split, ArrowDownToLine, ArrowRightToLine
} from 'lucide-vue-next'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])
const store = useEditorStore()
const isComposing = ref(false)
const md = new MarkdownIt({ html: true, breaks: true })
md.use(taskLists)

// 防抖保存
const debouncedSave = useDebounceFn(() => {
  // 触发保存逻辑 (通常在 Store 或父组件处理)
}, 1000)

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
  // 初始化时将 Markdown 解析为 HTML，确保 Tiptap 正确识别结构
  content: md.render(props.modelValue || ''),
  extensions: [
    StarterKit.configure({
      heading: { levels: [1, 2, 3] },
    }),
    // 关键修复: 确保 Markdown 扩展能解析粘贴的内容
    Markdown.configure({
      html: true,
      transformPastedText: true,
      transformCopiedText: true,
    }),
    Typography,
    Image,
    // 表格扩展
    Table.configure({ resizable: true }),
    TableRow,
    TableHeader,
    TableCell,
    TaskList,
    TaskItem.configure({
      nested: true,
    }),
  ],
  editorProps: {
    attributes: {
      // 样式修复: 增加 min-h, 确保点击区域够大
      // "Page" view styling: white background, shadow, padding
      class: 'prose prose-stone max-w-none focus:outline-none min-h-[calc(100vh-12rem)] px-12 py-10 bg-white shadow-sm mx-auto',
    },
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

// 关键修复: 监听 props 变化，实现双向绑定
watch(() => props.modelValue, (newValue) => {
  if (editor.value) {
    const currentMarkdown = (editor.value.storage as any).markdown.getMarkdown()
    if (newValue !== currentMarkdown) {
      // 外部更新时，重新解析并设置内容
      // 注意：这里简单全量替换可能会导致光标丢失，但对于文件切换是必须的
      editor.value.commands.setContent(md.render(newValue || ''))
    }
  }
})

const onCompositionStart = () => { isComposing.value = true }
const onCompositionEnd = () => { isComposing.value = false }

// Toolbar Actions
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
const insertTable = () => editor.value?.chain().focus().insertTable({ rows: 3, cols: 3, withHeaderRow: true }).run()

// Table Actions
const addColumnBefore = () => editor.value?.chain().focus().addColumnBefore().run()
const addColumnAfter = () => editor.value?.chain().focus().addColumnAfter().run()
const deleteColumn = () => editor.value?.chain().focus().deleteColumn().run()
const addRowBefore = () => editor.value?.chain().focus().addRowBefore().run()
const addRowAfter = () => editor.value?.chain().focus().addRowAfter().run()
const deleteRow = () => editor.value?.chain().focus().deleteRow().run()
const deleteTable = () => editor.value?.chain().focus().deleteTable().run()
const mergeCells = () => editor.value?.chain().focus().mergeCells().run()
const splitCell = () => editor.value?.chain().focus().splitCell().run()
const toggleHeaderColumn = () => editor.value?.chain().focus().toggleHeaderColumn().run()
const toggleHeaderRow = () => editor.value?.chain().focus().toggleHeaderRow().run()

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
        <button @click="insertTable" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="插入表格">
          <TableIcon class="w-4 h-4" />
        </button>
        <button @click="setHorizontalRule" class="p-1.5 rounded hover:bg-stone-100 text-stone-500" title="水平分割线">
          <Minus class="w-4 h-4" />
        </button>
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

    <!-- Text Formatting Bubble Menu -->
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

    <!-- Table Operations Bubble Menu -->
    <bubble-menu
      v-if="editor && editor.isActive('table')"
      :editor="editor"
      :tippy-options="{ duration: 100, placement: 'top' }"
      class="bg-white border border-stone-200 shadow-xl rounded-lg overflow-hidden flex flex-wrap max-w-sm"
    >
      <div class="flex divide-x divide-stone-100 border-b border-stone-100 w-full">
        <button @click="addColumnBefore" class="p-2 hover:bg-stone-50 text-stone-600" title="左侧插入列"><ArrowRightToLine class="w-4 h-4 rotate-180" /></button>
        <button @click="addColumnAfter" class="p-2 hover:bg-stone-50 text-stone-600" title="右侧插入列"><ArrowRightToLine class="w-4 h-4" /></button>
        <button @click="deleteColumn" class="p-2 hover:bg-red-50 text-red-600" title="删除列"><Trash2 class="w-4 h-4" /></button>
        <button @click="addRowBefore" class="p-2 hover:bg-stone-50 text-stone-600" title="上方插入行"><ArrowDownToLine class="w-4 h-4 rotate-180" /></button>
        <button @click="addRowAfter" class="p-2 hover:bg-stone-50 text-stone-600" title="下方插入行"><ArrowDownToLine class="w-4 h-4" /></button>
        <button @click="deleteRow" class="p-2 hover:bg-red-50 text-red-600" title="删除行"><Trash2 class="w-4 h-4" /></button>
      </div>
      <div class="flex divide-x divide-stone-100 w-full">
        <button @click="mergeCells" class="p-2 hover:bg-stone-50 text-stone-600" title="合并单元格"><Merge class="w-4 h-4" /></button>
        <button @click="splitCell" class="p-2 hover:bg-stone-50 text-stone-600" title="拆分单元格"><Split class="w-4 h-4" /></button>
        <button @click="toggleHeaderRow" :class="{ 'bg-stone-100': editor.isActive('tableHeader') }" class="p-2 hover:bg-stone-50 text-stone-600" title="表头行"><Rows class="w-4 h-4" /></button>
        <button @click="toggleHeaderColumn" class="p-2 hover:bg-stone-50 text-stone-600" title="表头列"><Columns class="w-4 h-4" /></button>
        <button @click="deleteTable" class="p-2 hover:bg-red-50 text-red-600 ml-auto" title="删除表格"><Trash2 class="w-4 h-4" /></button>
      </div>
    </bubble-menu>
  </div>
</template>

<style scoped>
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-thumb { background-color: #e5e5e5; border-radius: 3px; }

/* Table specific overrides for bubble menu icons */
.rotate-180 { transform: rotate(180deg); }
</style>