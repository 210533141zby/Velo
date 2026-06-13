import client from './client';
import type { DocumentRecord } from '@/types/document';

type DocumentDraft = Pick<DocumentRecord, 'title' | 'content'>;

/**
 * 获取文档列表。
 */
export function listDocuments() {
  return client.get<never, DocumentRecord[]>('/documents/');
}

/**
 * 获取文档。
 */
export function fetchDocument(documentId: number) {
  return client.get<never, DocumentRecord>(`/documents/${documentId}`);
}

/**
 * 创建一篇新文档。
 */
export function createDocument(payload: DocumentDraft) {
  return client.post<never, DocumentRecord>('/documents/', payload);
}

/**
 * 持久化指定文档的标题和正文。
 */
export function updateDocument(documentId: number, payload: DocumentDraft) {
  return client.put<never, Partial<DocumentRecord>>(`/documents/${documentId}`, payload);
}

/**
 * 删除指定文档。
 */
export function removeDocument(documentId: number) {
  return client.delete(`/documents/${documentId}`);
}
