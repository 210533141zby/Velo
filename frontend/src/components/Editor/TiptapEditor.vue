<script setup lang="ts">
/**
 * TiptapEditor.vue - 核心富文本编辑器组件
 * 
 * 功能摘要:
 * 本文件是前端编辑器的核心，集成了 Tiptap (一个基于 ProseMirror 的 Headless 编辑器框架)。
 * 它负责：
 * 1. 渲染编辑器 UI (工具栏 + 编辑区域)。
 * 2. 管理编辑器的扩展 (Markdown 支持, 任务列表等)。
 * 3. 处理复杂的交互逻辑 (如 粘贴 Markdown 源码自动解析)。
 * 4. 与 Pinia Store 同步文档内容 (双向绑定)。
 * 
 * 依赖关系:
 * - props.modelValue: 从父组件接收 Markdown 字符串。
 * - useEditorStore: 用于更新全局状态 (字数统计, 光标位置)。
 * - markdown-it: 用于在粘贴时辅助检测和解析 Markdown 语法。
 */
import { useEditor, EditorContent } from '@tiptap/vue-3'
import { Extension } from '@tiptap/core'
import StarterKit from '@tiptap/starter-kit'
import { Markdown } from 'tiptap-markdown'
import Typography from '@tiptap/extension-typography'
import Image from '@tiptap/extension-image'
import TaskList from '@tiptap/extension-task-list'
import TaskItem from '@tiptap/extension-task-item'
import { watch, onBeforeUnmount, ref } from 'vue'
import { useEditorStore } from '../../stores/editorStore'
import { useDebounceFn } from '@vueuse/core'
import MarkdownIt from 'markdown-it'
// @ts-ignore
import taskLists from 'markdown-it-task-lists'
import { 
  Bold, Italic, Strikethrough, Code, Heading1, Heading2, Heading3, 
  List, ListOrdered, Quote, Minus, CheckSquare, X
} from 'lucide-vue-next'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])
const store = useEditorStore()
const isComposing = ref(false) // 标记中文输入法状态，防止拼音输入过程中频繁触发更新
const md = new MarkdownIt({ html: true, breaks: true })
md.use(taskLists)

const debouncedSave = useDebounceFn(() => {
  // Save logic handled by store or parent usually
}, 1000)

/**
 * 更新编辑器统计信息
 * 
 * Logic Flow:
 * 1. 获取纯文本内容，计算字数 (简单的空格分割)。
 * 2. 获取当前光标位置 (Selection)。
 * 3. 计算光标所在的行号和列号 (通过统计光标前的换行符数量)。
 * 4. 调用 Store Action 更新状态，供底部状态栏显示。
 */
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
  content: props.modelValue, // 初始化内容
  extensions: [
    StarterKit.configure({ 
      // 开启所有标题级别支持 (H1-H6)，默认可能只有 H1-H3
      heading: { levels: [1, 2, 3, 4, 5, 6] }, 
    }),
    Markdown.configure({ 
      html: true, 
      transformPastedText: true, 
      transformCopiedText: true, 
    }), 
    Typography, 
    Image, 
    TaskList, 
    TaskItem.configure({ nested: true }),
  ],  
  editorProps: { 
    attributes: { 
      // Tailwind Typography (prose) 样式配置
      class: 'prose prose-stone max-w-none focus:outline-none min-h-[calc(100vh-12rem)] px-12 py-10 bg-white shadow-sm mx-auto', 
    }, 
    /**
     * handlePaste
     * 
     * Why:
     * 用户可能从 VS Code 或其他 Markdown 编辑器复制一段 Markdown 源码 (如 **Bold**)。
     * 默认粘贴可能会把它们当成纯文本，或者转义了特殊字符。
     * 我们需要拦截粘贴，如果发现内容像 Markdown，则强制渲染为 HTML 再插入。
     */
    handlePaste: (_view, event) => { 
      const text = event.clipboardData?.getData('text/plain') 
      const html = event.clipboardData?.getData('text/html') 
      
      // 如果剪贴板里有 HTML，通常优先用 HTML（保留样式）。 
      // 但如果用户明显是在粘贴一段 Markdown 源码...
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
    // 获取 Markdown 内容并更新父组件
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

// 监听 props.modelValue 变化 (双向绑定)
// 当父组件 (或 WebSocket) 更新内容时，同步到编辑器
watch(() => props.modelValue, (newValue) => { 
  if (editor.value) { 
    const currentMarkdown = (editor.value.storage as any).markdown.getMarkdown() 
    // 只有内容真正变了才 setContent，避免光标跳动
    if (newValue !== currentMarkdown) { 
      editor.value.commands.setContent(newValue) 
    } 
  } 
}) 

const onCompositionStart = () => { isComposing.value = true } 
const onCompositionEnd = () => { isComposing.value = false } 

// --- Toolbar Actions --- 
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

/**
 * toggleBlockquote (Smart)
 * 
 * Logic Flow:
 * 1. 检查是否是部分选中文本 (Partial Selection)。
 *    即: 选中了段落的一部分，而不是整个段落。
 * 2. 如果是部分选中:
 *    - 先在选区末尾 (End) 进行分割 (splitBlock)。
 *    - 再在选区开头 (Start) 进行分割 (splitBlock)。
 *    - 这样就把选中的文本“孤立”成了一个独立的 Block。
 *    - 最后对这个独立的 Block 执行 toggleBlockquote。
 * 3. 否则 (全选或光标模式):
 *    - 执行默认的 toggleBlockquote (包裹整个 Block)。
 * 
 * Why:
 * 用户反馈默认的 Blockquote 会把整段话都引用，体验不好。
 * 此逻辑实现了“选中哪句引用哪句”的直观效果。
 */
const toggleBlockquote = () => {
  if (!editor.value) return
  const { state } = editor.value
  const { selection } = state
  const { $from, $to, empty } = selection
  
  // 判断是否是单一段落内的部分选中
  const isPartial = !empty && 
                    $from.parent === $to.parent && 
                    $from.parent.isTextblock && 
                    ($from.parentOffset > 0 || $to.parentOffset < $from.parent.content.size)

  if (isPartial) {
    const chain = editor.value.chain().focus()
    
    // 1. 如果没选到末尾，先在末尾切一刀
    if ($to.parentOffset < $to.parent.content.size) {
      chain.setTextSelection($to.pos).splitBlock()
    }
    
    // 2. 如果没选到开头，再在开头切一刀
    if ($from.parentOffset > 0) {
      chain.setTextSelection($from.pos).splitBlock()
    } else {
      // 如果开头是顶格的，但刚才切了末尾，光标可能跑到了下一段，需要归位
      chain.setTextSelection($from.pos)
    }
    
    // 3. 对当前独立的 Block 进行引用
    chain.toggleBlockquote().run()
    return
  }

  // 默认行为
  editor.value.chain().focus().toggleBlockquote().run()
}

const setHorizontalRule = () => editor.value?.chain().focus().setHorizontalRule().run() 

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

  </div>
</template>

<style>
/* 全局样式覆盖 */
.custom-scrollbar::-webkit-scrollbar { width: 6px; }
.custom-scrollbar::-webkit-scrollbar-thumb { background-color: #e5e5e5; border-radius: 3px; }
.rotate-90 { transform: rotate(90deg); }
</style>