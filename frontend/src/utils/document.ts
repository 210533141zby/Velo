const SHANGHAI_FORMATTER = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai',
  hour12: false,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
});

/**
 * 把后端返回的时间字符串统一成浏览器可解析的 ISO 形式。
 */
function normalizeBackendDate(dateString: string) {
  if (dateString.includes('Z') || /[+-]\d{2}:\d{2}$/.test(dateString)) {
    return dateString;
  }

  return `${dateString.replace(' ', 'T')}Z`;
}

/**
 * 清理标题中的空白和默认占位文案。
 */
export function normalizeDocumentTitle(title?: string | null) {
  const normalized = (title ?? '').trim();
  const titleLower = normalized.toLowerCase();

  if (!normalized || titleLower === 'untitled' || titleLower === 'untitled document') {
    return '';
  }

  return normalized;
}

/**
 * 获取文档display标题。
 */
export function getDocumentDisplayTitle(title?: string | null) {
  return normalizeDocumentTitle(title) || 'Untitled';
}

/**
 * 把文档更新时间格式化为上海时区的展示文案。
 */
export function formatDocumentUpdatedAt(dateString?: string | null) {
  if (!dateString) {
    return '';
  }

  const date = new Date(normalizeBackendDate(dateString));
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  return SHANGHAI_FORMATTER.format(date);
}
