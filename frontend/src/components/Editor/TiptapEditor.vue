
<script setup lang="ts">
/**
 * TiptapEditor.vue - 核心富文本编辑器组件 (修复版)
 */
import { useEditor, EditorContent } from '@tiptap/vue-3'
import { Extension, Editor } from '@tiptap/core'
import { Plugin, PluginKey } from '@tiptap/pm/state'
import { Decoration, DecorationSet } from '@tiptap/pm/view'
import { EditorView } from '@tiptap/pm/view' // 补充导入
// @ts-ignore
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
  List, ListOrdered, Quote, Minus, CheckSquare
} from 'lucide-vue-next'
import axios from 'axios'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits(['update:modelValue'])
const store = useEditorStore()
const isComposing = ref(false)
const md = new MarkdownIt({ html: true, breaks: true })
md.use(taskLists)

// --- 1. Ghost Text (幽灵文本) 状态管理 ---
const ghostText = ref('')
const ghostPos = ref<number | null>(null)

// 独立的 API 调用函数
const fetchCompletion = async (prefix: string, suffix: string) => {
  try {
    const res = await axios.post('/api/v1/completion', {
      prefix,
      suffix,
      language: 'markdown'
    })
    return res.data.completion
  } catch (e) {
    console.error('[GhostDebug] 请求失败:', e)
    return ''
  }
}

// 独立的 Debounce Timer 变量
let completionTimer: ReturnType<typeof setTimeout> | null = null

// 核心补全执行逻辑 (复用)
const triggerCoreCompletion = async (editorInstance: Editor) => {
  if (isComposing.value || !editorInstance) return

  const { state } = editorInstance
  const currentFrom = state.selection.from
  
  // 上下文截取：前文 1000 字符，后文 200 字符
  const prefix = state.doc.textBetween(Math.max(0, currentFrom - 1000), currentFrom, '\n')
  const suffix = state.doc.textBetween(currentFrom, Math.min(state.doc.content.size, currentFrom + 200), '\n')

  const completion = await fetchCompletion(prefix, suffix)
  
  if (completion) {
    ghostText.value = completion
    ghostPos.value = currentFrom
    // 强制触发视图更新，渲染幽灵文字
    editorInstance.view.dispatch(editorInstance.state.tr)
  }
}

// 动态防抖补全触发逻辑 (统一处理入口)
const handleAutoCompletion = (editorInstance: Editor) => {
  if (isComposing.value || !editorInstance) return

  // 1. 清除旧定时器 (防抖核心)
  if (completionTimer) clearTimeout(completionTimer)

  // 2. 获取编辑器状态
  const { state } = editorInstance
  const { selection, doc } = state
  const { from } = selection

  // 3. 动态延迟判定
  // 检查光标后 50 个字符，如果全是空白，则认为在文末 (Append Mode)
  const suffixCheck = doc.textBetween(from, Math.min(doc.content.size, from + 50), '\n')
  const isAtEnd = !suffixCheck.trim()
  
  // NEW RULE: 只有文末才自动触发，文中必须手动触发
  if (!isAtEnd) return

  // 文末极速模式：250ms
  const delay = 250

  // 4. 启动新定时器
  completionTimer = setTimeout(() => {
    triggerCoreCompletion(editorInstance)
  }, delay)
}

// --- 2. Ghost Text Tiptap Extension ---
// 定义插件 Key
const ghostPluginKey = new PluginKey<DecorationSet>('ghostText')

const GhostTextExtension = Extension.create({
  name: 'ghostText',

  // 移除 addKeyboardShortcuts，改用 handleKeyDown 统一处理
  addProseMirrorPlugins() {
    // 捕获 Tiptap Editor 实例
    const editorInstance = this.editor

    return [
      new Plugin({
        key: ghostPluginKey,
        state: {
          init() {
            return DecorationSet.empty
          },
          apply(tr) {
            // console.log('[GhostDebug] Extension Apply 触发', { hasGhostText: !!ghostText.value, trDocChanged: tr.docChanged })
            // 每次文档变更，先清空幽灵文本 (除非是 debounce 刚回来)
            if (tr.docChanged || tr.selectionSet) {
               if (ghostText.value) {
                 ghostText.value = ''
                 ghostPos.value = null
                 return DecorationSet.empty
               }
            }
            
            // 如果 Vue Ref 里有值，就创建 Decoration
            if (ghostText.value && ghostPos.value !== null) {
               const widget = document.createElement('span')
               widget.className = 'ghost-text'
               widget.textContent = ghostText.value
               
               // 创建 Widget Decoration
               const deco = Decoration.widget(ghostPos.value, widget, {
                 side: 1
               })
               return DecorationSet.create(tr.doc, [deco])
            }
            
            return DecorationSet.empty
          }
        },
        props: {
          decorations(state) {
            return ghostPluginKey.getState(state)
          },
          handleKeyDown(view: EditorView, event: KeyboardEvent) {
            // 1. 手动触发快捷键 (Mod+Alt / Ctrl+Alt / Mod+Space)
            const isMod = event.ctrlKey || event.metaKey
            const isAlt = event.altKey
            
            // 方案 A: Mod + Alt
            if (isMod && isAlt) {
              console.log('[GhostDebug] Manual Trigger: Mod+Alt')
              event.preventDefault()
              triggerCoreCompletion(editorInstance)
              return true
            }

            // 方案 B: Mod + Space (备选)
            if (isMod && event.code === 'Space') {
              console.log('[GhostDebug] Manual Trigger: Mod+Space')
              event.preventDefault()
              triggerCoreCompletion(editorInstance)
              return true
            }

            // 2. 拦截 Tab 键：采纳
            if (event.key === 'Tab' && ghostText.value && ghostPos.value !== null) {
              event.preventDefault()
              
              // 插入真实文本
              const tr = view.state.tr.insertText(ghostText.value, ghostPos.value)
              view.dispatch(tr)
              
              // 清空状态
              ghostText.value = ''
              ghostPos.value = null
              return true
            }
            
            // 3. 拦截 Esc 键：拒绝
            if (event.key === 'Escape' && ghostText.value) {
              event.preventDefault()
              ghostText.value = ''
              ghostPos.value = null
              view.dispatch(view.state.tr)
              return true
            }
            
            return false
          }
        }
      })
    ]
  }
})

// --- 3. Markdown Paste Logic Extension (修复循环引用) ---
const MarkdownPasteLogic = Extension.create({
  name: 'markdownPasteLogic',
  
  addProseMirrorPlugins() {
    // 关键修复：使用 (this as any) 绕过 TS 检查
    const editor = (this as any).editor as Editor

    return [
      new Plugin({
        key: new PluginKey('markdownPaste'),
        props: {
          handlePaste: (_view, event) => {
            const text = event.clipboardData?.getData('text/plain')
            const html = event.clipboardData?.getData('text/html')

            if (text) {
              // 特征检测 Markdown
              const isMarkdownLike = /^(\s*#{1,6}\s|\s*>|\s*[-*]\s|\s*\d+\.\s|`{3})/.test(text)

              if (isMarkdownLike && (!html || html.length < text.length * 1.5)) {
                const parsedHtml = md.render(text)
                editor.commands.insertContent(parsedHtml)
                return true
              }
            }
            return false
          }
        }
      })
    ]
  }
})

const debouncedSave = useDebounceFn(() => {
  // Save logic
}, 1000)

const updateStats = (editorInstance: Editor) => {
  const text = editorInstance.state.doc.textContent
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0
  
  const selection = editorInstance.state.selection
  const { from } = selection
  const textBefore = editorInstance.state.doc.textBetween(0, from, '\n')
  const lines = textBefore.split('\n')
  const line = lines.length
  const col = lines[lines.length - 1].length + 1
  
  store.updateStats(wordCount, line, col)
}

// --- 初始化编辑器 ---
const editor = useEditor({
  content: props.modelValue,
  extensions: [
    StarterKit.configure({ 
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
    GhostTextExtension,  // 注册 AI 补全插件
    MarkdownPasteLogic,  // 注册粘贴逻辑插件
  ],  
  editorProps: { 
    attributes: { 
      class: 'prose prose-stone max-w-none focus:outline-none min-h-[calc(100vh-12rem)] px-12 py-10 bg-white shadow-sm mx-auto', 
    }, 
  }, 
  onUpdate: ({ editor }) => { 
    // 1. 无论是否在输入中文，先打印日志证明函数活着 
    console.log('[GhostDebug] onUpdate 触发', { isComposing: isComposing.value }); 
 
    // 2. 强制执行补全触发器 (移到 isComposing 检查之前!) 
    handleAutoCompletion(editor); 
 
    // 3. 原有的数据同步逻辑保持不变 
    if (isComposing.value) return; 
    const markdown = (editor.storage as any).markdown.getMarkdown() 
    emit('update:modelValue', markdown) 
    store.updateContent(markdown) 
    debouncedSave() 
    updateStats(editor)
  }, 
  onSelectionUpdate: ({ editor }) => {
    // 光标移动也触发补全逻辑 (关键新增)
    handleAutoCompletion(editor)
    updateStats(editor)
  },
}) 

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
const toggleBlockquote = () => {
  if (!editor.value) return
  const { state } = editor.value
  const { selection } = state
  const { $from, $to, empty } = selection
  
  const isPartial = !empty && 
                    $from.parent === $to.parent && 
                    $from.parent.isTextblock && 
                    ($from.parentOffset > 0 || $to.parentOffset < $from.parent.content.size)

  if (isPartial) {
    const chain = editor.value.chain().focus()
    if ($to.parentOffset < $to.parent.content.size) chain.setTextSelection($to.pos).splitBlock()
    if ($from.parentOffset > 0) chain.setTextSelection($from.pos).splitBlock()
    else chain.setTextSelection($from.pos)
    chain.toggleBlockquote().run()
    return
  }
  editor.value.chain().focus().toggleBlockquote().run()
}
const setHorizontalRule = () => editor.value?.chain().focus().setHorizontalRule().run() 

onBeforeUnmount(() => editor.value?.destroy())
</script>

<template>
  <div class="flex flex-col h-full w-full bg-[#F9F9F9] relative overflow-hidden">
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

/* Ghost Text (幽灵文本) 样式 */
.ghost-text {
  color: #adb5bd; /* 灰色 */
  font-style: italic;
  pointer-events: none;
}
/* Tab 提示 */
.ghost-text::after {
  content: 'Tab';
  font-size: 0.7em;
  background: #e9ecef;
  border-radius: 3px;
  padding: 0 4px;
  margin-left: 4px;
  color: #6c757d;
  vertical-align: middle;
}
</style>