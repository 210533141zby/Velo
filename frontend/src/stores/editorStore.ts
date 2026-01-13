import { defineStore } from 'pinia';
import { ref } from 'vue';
import axios from 'axios';
import { useDebounceFn } from '@vueuse/core';

const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
});

export interface Document {
  id: number;
  title: string;
  content: string;
  created_at: string;
  updated_at: string;
}

export type SaveStatus = 'saved' | 'saving' | 'error' | 'unsaved';

export const useEditorStore = defineStore('editor', () => {
  // State
  const documents = ref<Document[]>([]);
  const currentDocument = ref<Document | null>(null);
  const saveStatus = ref<SaveStatus>('saved');
  const isSidebarOpen = ref(true);
  const isCopilotOpen = ref(false);
  const wordCount = ref(0);
  const cursorPosition = ref({ line: 1, col: 1 });
  const isAiThinking = ref(false);

  // Actions
  async function fetchDocuments() {
    try {
      const res = await api.get('/documents/');
      documents.value = res.data;
      if (!currentDocument.value && documents.value.length > 0) {
        await loadDocument(documents.value[0].id);
      }
    } catch (e) {
      console.error('Failed to fetch documents', e);
    }
  }

  async function createDocument() {
    try {
      const res = await api.post('/documents/', {
        title: '',
        content: '',
      });
      const newDoc = res.data as Document;
      // Force title to be empty if it comes back as Untitled or null (defensive fix)
      // We check for various forms of 'Untitled' to be safe
      const titleLower = (newDoc.title || '').toLowerCase().trim();
      if (!newDoc.title || titleLower === 'untitled' || titleLower === 'untitled document') {
        newDoc.title = '';
      }
      documents.value.unshift(newDoc);
      currentDocument.value = newDoc;
      saveStatus.value = 'saved';
    } catch (e) {
      console.error('Failed to create document', e);
    }
  }

  async function loadDocument(id: number) {
    try {
      const res = await api.get(`/documents/${id}`);
      currentDocument.value = res.data as Document;
      saveStatus.value = 'saved';
    } catch (e) {
      console.error('Failed to load document', e);
    }
  }

  async function deleteDocument(id: number) {
    try {
      await api.delete(`/documents/${id}`);
      const idx = documents.value.findIndex(d => d.id === id);
      if (idx !== -1) {
        documents.value.splice(idx, 1);
      }
      
      // If deleted current document, switch to another one
      if (currentDocument.value?.id === id) {
        if (documents.value.length > 0) {
          await loadDocument(documents.value[0].id);
        } else {
          currentDocument.value = null;
        }
      }
    } catch (e) {
      console.error('Failed to delete document', e);
    }
  }

  async function saveCurrentDocument() {
    if (!currentDocument.value) return;
    saveStatus.value = 'saving';
    try {
      const res = await api.put(`/documents/${currentDocument.value.id}`, {
        title: currentDocument.value.title,
        content: currentDocument.value.content,
      });
      currentDocument.value = { ...currentDocument.value, ...(res.data as Partial<Document>) };
      
      const idx = documents.value.findIndex(d => d.id === currentDocument.value?.id);
      if (idx !== -1) documents.value[idx] = currentDocument.value;
      
      saveStatus.value = 'saved';
    } catch (e) {
      saveStatus.value = 'error';
      console.error('Failed to save', e);
    }
  }

  const debouncedSave = useDebounceFn(() => {
    saveCurrentDocument();
  }, 2000);

  function updateContent(content: string) {
    if (!currentDocument.value) return;
    currentDocument.value.content = content;
    saveStatus.value = 'unsaved'; // Set to unsaved immediately on change
    debouncedSave();
  }

  function updateTitle(title: string) {
    if (!currentDocument.value) return;
    currentDocument.value.title = title;
    saveStatus.value = 'unsaved';
    debouncedSave();
  }

  function toggleSidebar() {
    isSidebarOpen.value = !isSidebarOpen.value;
  }

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
    fetchDocuments,
    createDocument,
    loadDocument,
    saveCurrentDocument,
    deleteDocument,
    updateContent,
    updateTitle,
    toggleSidebar,
    toggleCopilot
  };
});
