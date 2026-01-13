<script setup lang="ts">
import { ref, watch } from 'vue';
import { getAllFolders, moveDocument, type Doc, type Folder } from '@/api';
import { ElMessage } from 'element-plus';

const props = defineProps<{
  visible: boolean;
  doc: Doc | null;
}>();

const emit = defineEmits<{
  (e: 'update:visible', value: boolean): void;
  (e: 'success'): void;
}>();

const allFolders = ref<Folder[]>([]);
const moveTargetFolderId = ref<number | null>(null);

watch(() => props.visible, async (val) => {
  if (val) {
    moveTargetFolderId.value = null;
    try {
      allFolders.value = await getAllFolders();
    } catch (e) {
      console.error(e);
    }
  }
});

const handleMove = async () => {
  if (!props.doc) return;
  const targetId = moveTargetFolderId.value === null ? 0 : moveTargetFolderId.value;
  
  try {
    await moveDocument(props.doc.id!, targetId);
    ElMessage.success('文档已移动');
    emit('success');
    emit('update:visible', false);
  } catch (e) {
    // Error handled
  }
};
</script>

<template>
  <div v-if="visible" class="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur-sm">
    <div class="bg-white rounded-xl p-6 w-96 shadow-2xl">
      <h3 class="text-lg font-bold mb-4 text-gray-900">移动文档</h3>
      <p class="mb-3 text-sm text-gray-500 font-medium">将 "{{ doc?.title }}" 移动到:</p>
      <select v-model="moveTargetFolderId" class="w-full border border-gray-300 rounded-lg px-4 py-2.5 mb-6 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 text-sm bg-white">
        <option :value="null">根目录</option>
        <option v-for="folder in allFolders" :key="folder.id" :value="folder.id">
          {{ folder.title }}
        </option>
      </select>
      <div class="flex justify-end gap-3">
        <button @click="emit('update:visible', false)" class="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg font-medium text-sm transition-colors">取消</button>
        <button @click="handleMove" class="px-4 py-2 bg-primary text-white rounded-lg hover:opacity-90 font-medium text-sm shadow-sm transition-all">移动</button>
      </div>
    </div>
  </div>
</template>
