import request from './request';

export interface Doc {
  id?: number;
  title: string;
  content: string;
  folder_id?: number | null;
  created_at?: string;
  updated_at?: string;
}

export interface Folder {
  id: number;
  title: string;
  parent_id?: number | null;
}

export const getFolderContents = (folderId: number) => {
  return request.get(`/folders/${folderId}/contents`) as Promise<{ documents: Doc[]; folders: Folder[] }>;
};

export const createFolder = (data: { title: string; parent_id: number | null }) => {
  return request.post('/folders/', data) as Promise<Folder>;
};

export const getAllFolders = () => {
  return request.get('/folders/all') as Promise<Folder[]>;
};

export const saveDocument = (doc: Doc) => {
  if (doc.id) {
    return request.put(`/documents/${doc.id}`, doc) as Promise<Doc>;
  } else {
    return request.post('/documents/', doc) as Promise<Doc>;
  }
};

export const deleteDocument = (docId: number) => {
  return request.delete(`/documents/${docId}`);
};

export const moveDocument = (docId: number, folderId: number) => {
  return request.put(`/documents/${docId}`, { folder_id: folderId });
};

export const chatWithAgent = (query: string, useRAG: boolean = false) => {
  const endpoint = useRAG ? '/agent/rag_chat' : '/agent/chat';
  return request.post(endpoint, { query }) as Promise<{ response: string }>;
};

export const polishText = (text: string) => {
  return request.post('/agent/polish', { text }) as Promise<{ polished_text: string }>;
};

export const uploadImage = (formData: FormData) => {
  return request.post('/images/upload', formData) as Promise<{ url: string }>;
};

export const codeCompletion = (data: { prefix: string; suffix: string; max_tokens: number }) => {
  return request.post('/code/completion', data) as Promise<{ completion: string }>;
};
