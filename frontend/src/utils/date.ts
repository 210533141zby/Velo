export const formatDate = (dateString: string | undefined | null) => {
  if (!dateString) return '';
  
  // 解决时区问题：如果后端返回的是 UTC 时间但没有带时区后缀 (Z 或 +00:00)
  let dateStr = dateString;
  if (typeof dateStr === 'string' && !dateStr.includes('Z') && !dateStr.includes('+')) {
      if (dateStr.includes('T')) {
          dateStr += 'Z';
      } else {
           dateStr = dateStr.replace(' ', 'T') + 'Z';
      }
  }

  const date = new Date(dateStr);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hour = String(date.getHours()).padStart(2, '0');
  const minute = String(date.getMinutes()).padStart(2, '0');
  const second = String(date.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
};
