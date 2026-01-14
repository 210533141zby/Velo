<script setup lang="ts">
/**
 * Editor.vue
 * 
 * 核心编辑器组件
 * 集成 Monaco Editor 和 AI 代码补全 (Copilot 模式)
 * 支持 Markdown 语法高亮、实时 AI 补全、中文输入优化
 */
import { onMounted, onUnmounted, ref, shallowRef } from 'vue';
import * as monaco from 'monaco-editor';
import { codeCompletion } from '@/api';

// 编辑器容器引用
const editorContainer = ref<HTMLElement | null>(null);
// Monaco 实例引用 (使用 shallowRef 避免 Vue 深度响应式导致的性能问题)
const editorRef = shallowRef<monaco.editor.IStandaloneCodeEditor | null>(null);
// AI 补全 Provider 引用，用于后续销毁
const completionProvider = shallowRef<monaco.IDisposable | null>(null);

// 中文输入法状态标记
let isComposing = false;
// 请求中断控制器
let abortController: AbortController | null = null;

/**
 * 初始化编辑器
 */
onMounted(() => {
  if (editorContainer.value) {
    // 创建编辑器实例
    editorRef.value = monaco.editor.create(editorContainer.value, {
      value: '# Welcome to Velo AI\n\nStart typing to see AI completions...',
      language: 'markdown',
      theme: 'vs-light', // 浅色主题
      fontSize: 14,
      fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
      lineHeight: 1.6,
      minimap: { enabled: false }, // 关闭缩略图
      wordWrap: 'on', // 自动换行
      padding: { top: 20, bottom: 20 },
      scrollBeyondLastLine: false,
      smoothScrolling: true,
      cursorBlinking: 'smooth',
      cursorSmoothCaretAnimation: 'on',
      // 开启内联建议 (Copilot 核心配置)
      inlineSuggest: {
        enabled: true,
        mode: 'subwordSmart',
      },
      // 隐藏部分不需要的 UI
      overviewRulerLanes: 0,
      hideCursorInOverviewRuler: true,
      scrollbar: {
        vertical: 'visible',
        horizontal: 'hidden',
        useShadows: false,
        verticalScrollbarSize: 8,
      },
    });

    // 监听中文输入状态 (中文输入过程中不触发 AI)
    const container = editorContainer.value;
    // 注意：需要监听 textarea 的事件，或者容器的捕获阶段
    container.addEventListener('compositionstart', () => { isComposing = true; }, true);
    container.addEventListener('compositionend', () => { isComposing = false; }, true);

    // 注册 AI 补全 Provider
    registerCopilotProvider();

    // 监听内容变化 (可选：用于自动保存等)
    editorRef.value.onDidChangeModelContent(() => {
      // console.log('Content changed');
    });
  }
});

/**
 * 销毁编辑器及相关资源
 */
onUnmounted(() => {
  if (completionProvider.value) {
    completionProvider.value.dispose();
  }
  if (editorRef.value) {
    editorRef.value.dispose();
  }
});

/**
 * 注册 Copilot 风格的内联补全 Provider
 */
const registerCopilotProvider = () => {
  completionProvider.value = monaco.languages.registerInlineCompletionsProvider('markdown', {
    provideInlineCompletions: async (model, position, _context, token) => {
      // 1. 如果正在输入中文，直接返回空
      if (isComposing) {
        return { items: [] };
      }

      // 2. 防抖 (简单的 sleep 实现，避免每个击键都请求)
      // Monaco 自身有触发间隔，但这里加一点延迟更稳妥
      await new Promise(resolve => setTimeout(resolve, 300));
      if (token.isCancellationRequested) return { items: [] };

      // 3. 中断上一次未完成的请求
      if (abortController) {
        abortController.abort();
      }
      abortController = new AbortController();

      try {
        // 4. 获取上下文
        // 获取光标前的所有文本 (限制长度避免 Token 溢出)
        const textUntilPosition = model.getValueInRange({
          startLineNumber: 1,
          startColumn: 1,
          endLineNumber: position.lineNumber,
          endColumn: position.column
        });

        // 获取光标后的文本
        const textAfterPosition = model.getValueInRange({
          startLineNumber: position.lineNumber,
          startColumn: position.column,
          endLineNumber: model.getLineCount(),
          endColumn: model.getLineMaxColumn(model.getLineCount())
        });

        // 截取上下文 (前 2000 字符，后 500 字符)
        const prefix = textUntilPosition.slice(-2000);
        const suffix = textAfterPosition.slice(0, 500);

        // 简单判断是否在 Python 代码块中，调整 Max Tokens
        // 这是一个简单的 heuristic
        const lastCodeBlock = prefix.lastIndexOf('```');
        const isPython = lastCodeBlock !== -1 && prefix.slice(lastCodeBlock).includes('python');
        const maxTokens = isPython ? 128 : 64;

        // 5. 调用后端 API
        // 注意：这里没有传递 signal 给 axios，因为 axios 取消需要 CancelToken 或 signal
        // 如果 API 库支持 signal 最好，否则这里只是逻辑上的“新请求覆盖旧请求”
        // 实际上，由于 JS 单线程，await 期间 abortController 可能会被下一次调用更新
        // 但我们在 await 之前已经更新了 controller。
        // 真正的网络取消需要修改 api/index.ts 支持 signal。
        // 这里暂不实现网络层取消，只做逻辑层丢弃。
        
        const res = await codeCompletion({
          prefix,
          suffix,
          max_tokens: maxTokens
        });

        // 如果 Token 已经被取消 (用户继续输入了)，则丢弃结果
        if (token.isCancellationRequested) return { items: [] };

        if (!res.completion) return { items: [] };

        // 6. 返回补全结果
        return {
          items: [{
            insertText: res.completion,
            range: {
              startLineNumber: position.lineNumber,
              startColumn: position.column,
              endLineNumber: position.lineNumber,
              endColumn: position.column
            }
          }]
        };

      } catch (error) {
        console.error('AI Completion Failed:', error);
        return { items: [] };
      }
    },
    freeInlineCompletions() {
      // 释放资源
    }
  });
};
</script>

<template>
  <div class="w-full h-full relative group bg-white">
    <!-- 编辑器挂载点 -->
    <div ref="editorContainer" class="w-full h-full min-h-[80vh]"></div>
    
    <!-- 状态指示器 -->
    <div class="absolute bottom-4 right-4 text-xs text-zinc-400 bg-white/80 backdrop-blur border border-zinc-100 px-3 py-1.5 rounded-full shadow-sm opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 pointer-events-none">
      <div class="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
      Velo AI Ready
    </div>
  </div>
</template>

<style>
/* 自定义 Monaco Editor 样式覆盖 */
.monaco-editor .margin {
  background-color: transparent !important;
}
</style>
