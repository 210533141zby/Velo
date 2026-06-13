import { defineStore } from 'pinia';
import { ref } from 'vue';
import { createDocument, fetchDocument, listDocuments, removeDocument, updateDocument } from '@/api/documents';
import type { DocumentRecord, SaveStatus } from '@/types/document';
import { normalizeDocumentTitle } from '@/utils/document';

type CursorPosition = {
  line: number;
  col: number;
};

type CompletionStatusTone = 'idle' | 'info' | 'success' | 'settled' | 'warning' | 'error';

const AUTO_SAVE_DELAY_MS = 2000;

/**
 * 统一文档对象中的标题格式，避免空标题和占位标题进入状态。
 */
function sanitizeDocument(document: DocumentRecord): DocumentRecord {
  return {
    ...document,
    title: normalizeDocumentTitle(document.title),
  };
}

/**
 * 创建并返回编辑器状态仓库。
 */
export const useEditorStore = defineStore('editor', () => {
  const documents = ref<DocumentRecord[]>([]);
  const currentDocument = ref<DocumentRecord | null>(null);
  const saveStatus = ref<SaveStatus>('saved');
  const isSidebarOpen = ref(true);
  const isCopilotOpen = ref(false);
  const wordCount = ref(0);
  const cursorPosition = ref<CursorPosition>({ line: 1, col: 1 });
  const isAiThinking = ref(false);
  const completionStatusMessage = ref('补全就绪');
  const completionStatusTone = ref<CompletionStatusTone>('idle');

  let saveTimer: number | null = null;
  const loadedDocumentIds = new Set<number>();
  const loadingDocumentPromises = new Map<number, Promise<DocumentRecord | null>>();

  /**
   * 清除当前排队中的自动保存定时器。
   */
  function clearScheduledSave() {
    if (saveTimer) {
      window.clearTimeout(saveTimer);
      saveTimer = null;
    }
  }

  /**
   * 按延迟策略重新安排一次自动保存。
   */
  function queueSave() {
    clearScheduledSave();
    saveTimer = window.setTimeout(() => {
      saveTimer = null;
      void saveCurrentDocument();
    }, AUTO_SAVE_DELAY_MS);
  }

  /**
   * 用最新文档内容更新列表中的摘要项；不存在时直接追加。
   */
  function syncDocumentInList(document: DocumentRecord) {
    const normalized = sanitizeDocument(document);
    const index = documents.value.findIndex((item) => item.id === normalized.id);

    if (index === -1) {
      documents.value.push(normalized);
    } else {
      documents.value[index] = normalized;
    }

    return normalized;
  }

  /**
   * 记录该文档已加载过完整内容，后续列表刷新时可保留本地正文。
   */
  function markDocumentAsLoaded(documentId: number) {
    loadedDocumentIds.add(documentId);
  }

  /**
   * 切换当前编辑中的文档，并重置待保存计时。
   */
  function selectDocument(document: DocumentRecord | null) {
    clearScheduledSave();
    currentDocument.value = document ? sanitizeDocument(document) : null;
  }

  /**
   * 获取文档明细。
   */
  async function fetchDocumentDetail(documentId: number, options?: { select?: boolean }) {
    const shouldSelect = options?.select ?? false;
    const cachedDocument = documents.value.find((document) => document.id === documentId);

    if (cachedDocument && loadedDocumentIds.has(documentId)) {
      if (shouldSelect) {
        selectDocument(cachedDocument);
        saveStatus.value = 'saved';
      }
      return cachedDocument;
    }

    const existingRequest = loadingDocumentPromises.get(documentId);
    if (existingRequest) {
      const document = await existingRequest;
      if (shouldSelect && document) {
        selectDocument(document);
        saveStatus.value = 'saved';
      }
      return document;
    }

    const request = (async () => {
      try {
        const document = syncDocumentInList(await fetchDocument(documentId));
        markDocumentAsLoaded(document.id);
        return document;
      } catch (error) {
        console.error('Failed to load document', error);
        return null;
      } finally {
        loadingDocumentPromises.delete(documentId);
      }
    })();

    loadingDocumentPromises.set(documentId, request);
    const document = await request;

    if (shouldSelect && document) {
      selectDocument(document);
      saveStatus.value = 'saved';
    }

    return document;
  }

  /**
   * 后台预取一批文档详情，减少首次切换时的等待。
   */
  async function prefetchDocumentDetails(documentIds: number[]) {
    const pendingIds = documentIds.filter(
      (documentId) => !loadedDocumentIds.has(documentId) && !loadingDocumentPromises.has(documentId),
    );

    if (!pendingIds.length) {
      return;
    }

    await Promise.all(pendingIds.map((documentId) => fetchDocumentDetail(documentId)));
  }

  /**
   * 获取文档列表。
   */
  async function fetchDocuments() {
    try {
      const existingById = new Map(documents.value.map((document) => [document.id, document]));
      const nextDocuments = (await listDocuments()).map((document) => {
        const normalized = sanitizeDocument(document);
        const cached = existingById.get(normalized.id);
        if (cached && loadedDocumentIds.has(normalized.id)) {
          return sanitizeDocument({
            ...normalized,
            content: cached.content,
          });
        }
        return normalized;
      });
      documents.value = nextDocuments;

      if (!nextDocuments.length) {
        selectDocument(null);
        return;
      }

      if (!currentDocument.value) {
        const initialDocument = await loadDocument(nextDocuments[0].id);
        const initialId = initialDocument?.id ?? nextDocuments[0].id;
        void prefetchDocumentDetails(nextDocuments.map((document) => document.id).filter((id) => id !== initialId));
        return;
      }

      const currentSummary = nextDocuments.find((document) => document.id === currentDocument.value?.id);
      if (!currentSummary) {
        const fallbackDocument = await loadDocument(nextDocuments[0].id);
        const fallbackId = fallbackDocument?.id ?? nextDocuments[0].id;
        void prefetchDocumentDetails(nextDocuments.map((document) => document.id).filter((id) => id !== fallbackId));
        return;
      }

      currentDocument.value = {
        ...currentDocument.value,
        ...currentSummary,
      };

      void prefetchDocumentDetails(
        nextDocuments.map((document) => document.id).filter((id) => id !== currentDocument.value?.id),
      );
    } catch (error) {
      console.error('Failed to fetch documents', error);
    }
  }

  /**
   * 创建空白文档，并立即切换到该文档开始编辑。
   */
  async function createNewDocument() {
    try {
      const newDocument = sanitizeDocument(
        await createDocument({
          title: '',
          content: '',
        }),
      );

      documents.value.unshift(newDocument);
      markDocumentAsLoaded(newDocument.id);
      selectDocument(newDocument);
      saveStatus.value = 'saved';
      return newDocument;
    } catch (error) {
      console.error('Failed to create document', error);
      return null;
    }
  }

  /**
   * 加载文档。
   */
  async function loadDocument(documentId: number) {
    if (currentDocument.value?.id === documentId) {
      return currentDocument.value;
    }

    return fetchDocumentDetail(documentId, { select: true });
  }

  /**
   * 删除指定文档；如果删掉的是当前文档，则自动切到下一篇。
   */
  async function deleteDocument(documentId: number) {
    try {
      await removeDocument(documentId);
      loadedDocumentIds.delete(documentId);
      documents.value = documents.value.filter((document) => document.id !== documentId);

      if (currentDocument.value?.id !== documentId) {
        return;
      }

      if (!documents.value.length) {
        selectDocument(null);
        return;
      }

      await loadDocument(documents.value[0].id);
    } catch (error) {
      console.error('Failed to delete document', error);
    }
  }

  /**
   * 保存当前文档；若保存期间又发生编辑，则继续排队下一次保存。
   */
  async function saveCurrentDocument() {
    if (!currentDocument.value) {
      return;
    }

    clearScheduledSave();

    const snapshot = {
      ...currentDocument.value,
    };

    saveStatus.value = 'saving';

    try {
      const response = await updateDocument(snapshot.id, {
        title: snapshot.title,
        content: snapshot.content,
      });

      const persistedDocument = sanitizeDocument({
        ...snapshot,
        ...response,
        title: response.title ?? snapshot.title,
        content: response.content ?? snapshot.content,
      });

      if (currentDocument.value?.id === snapshot.id) {
        const hasPendingLocalChanges =
          currentDocument.value.title !== snapshot.title || currentDocument.value.content !== snapshot.content;

        const nextCurrentDocument = hasPendingLocalChanges
          ? sanitizeDocument({
              ...currentDocument.value,
              updated_at: persistedDocument.updated_at,
              created_at: persistedDocument.created_at,
              folder_id: persistedDocument.folder_id,
            })
          : persistedDocument;

        currentDocument.value = nextCurrentDocument;
        syncDocumentInList(nextCurrentDocument);
        markDocumentAsLoaded(nextCurrentDocument.id);
        saveStatus.value = hasPendingLocalChanges ? 'unsaved' : 'saved';

        if (hasPendingLocalChanges) {
          queueSave();
        }

        return;
      }

      syncDocumentInList(persistedDocument);
      markDocumentAsLoaded(persistedDocument.id);
    } catch (error) {
      if (currentDocument.value?.id === snapshot.id) {
        saveStatus.value = 'error';
      }
      console.error('Failed to save document', error);
    }
  }

  /**
   * 更新正文并触发自动保存。
   */
  function updateContent(content: string) {
    if (!currentDocument.value || currentDocument.value.content === content) {
      return;
    }

    currentDocument.value.content = content;
    saveStatus.value = 'unsaved';
    queueSave();
  }

  /**
   * 更新标题并触发自动保存。
   */
  function updateTitle(title: string) {
    if (!currentDocument.value || currentDocument.value.title === title) {
      return;
    }

    currentDocument.value.title = title;
    saveStatus.value = 'unsaved';
    queueSave();
  }

  /**
   * 同步状态栏需要的字数和光标位置。
   */
  function updateStats(count: number, line: number, col: number) {
    wordCount.value = count;
    cursorPosition.value = { line, col };
  }

  /**
   * 更新 AI 处理中状态，用于顶部和底部状态提示。
   */
  function setAiThinking(nextValue: boolean) {
    isAiThinking.value = nextValue;
  }

  /**
   * 更新底部状态栏中的补全文案和状态色。
   */
  function setCompletionStatus(message: string, tone: CompletionStatusTone = 'info') {
    completionStatusMessage.value = message;
    completionStatusTone.value = tone;
  }

  /**
   * 切换左侧文档栏的显示状态。
   */
  function toggleSidebar() {
    isSidebarOpen.value = !isSidebarOpen.value;
  }

  /**
   * 切换知识库问答侧栏的显示状态。
   */
  function toggleCopilot() {
    isCopilotOpen.value = !isCopilotOpen.value;
  }

  return {
    documents,
    currentDocument,
    saveStatus,
    isSidebarOpen,
    isCopilotOpen,
    wordCount,
    cursorPosition,
    isAiThinking,
    completionStatusMessage,
    completionStatusTone,
    fetchDocuments,
    createDocument: createNewDocument,
    loadDocument,
    saveCurrentDocument,
    deleteDocument,
    updateContent,
    updateTitle,
    updateStats,
    setAiThinking,
    setCompletionStatus,
    toggleSidebar,
    toggleCopilot,
  };
});
