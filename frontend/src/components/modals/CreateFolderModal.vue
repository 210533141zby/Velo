<script setup lang="ts">
import { ref, watch } from 'vue';
import { createFolder } from '@/api';
import { ElMessage } from 'element-plus';

const props = defineProps<{
  visible: boolean;
  parentId: number | null;
}>();

const emit = defineEmits<{
  (e: 'update:visible', value: boolean): void;
  (e: 'success'): void;
}>();

const newFolderName = ref('');

watch(() => props.visible, (val) => {
  if (val) newFolderName.value = '';
});

const handleCreate = async () => {
  if (!newFolderName.value.trim()) return;
  
  try {
    await createFolder({
      title: newFolderName.value,
      parent_id: props.parentId
    });
    ElMessage.success('文件夹已创建');
    emit('success');
    emit('update:visible', false);
  } catch (e) {
    // Error handled in request interceptor
  }
};
</script>

<template>
  <div v-if="visible" class="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur-sm">
    <div class="bg-white rounded-xl p-6 w-96 shadow-2xl transform transition-all scale-100">
      <h3 class="text-lg font-bold mb-4 text-gray-900">新建文件夹</h3>
      <input v-model="newFolderName" @keyup.enter="handleCreate" type="text" placeholder="文件夹名称" class="w-full border border-gray-300 rounded-lg px-4 py-2.5 mb-6 focus:outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 text-sm">
      <div class="flex justify-end gap-3">
        <button @click="emit('update:visible', false)" class="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-lg font-medium text-sm transition-colors">取消</button>
        <button @click="handleCreate" class="px-4 py-2 bg-primary text-white rounded-lg hover:opacity-90 font-medium text-sm shadow-sm transition-all">创建</button>
      </div>
    </div>
  </div>
</template>
