<script setup lang="ts">
defineProps<{
  visible: boolean;
  message: string;
}>();

const emit = defineEmits<{
  (e: 'update:visible', value: boolean): void;
  (e: 'confirm'): void;
}>();
</script>

<template>
  <div v-if="visible" class="fixed inset-0 bg-black/60 flex items-center justify-center z-[60] backdrop-blur-sm">
    <div class="bg-surface rounded-xl p-6 w-96 shadow-2xl border border-neutral-100">
      <h3 class="text-lg font-bold mb-4 text-text-main font-serif">确认</h3>
      <p class="mb-6 text-text-sub font-medium">{{ message }}</p>
      <div class="flex justify-end gap-3">
        <button @click="emit('update:visible', false)" class="px-4 py-2 text-text-sub hover:bg-neutral-50 rounded-lg font-medium text-sm transition-colors">取消</button>
        <button @click="emit('confirm')" class="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-hover font-medium text-sm shadow-sm transition-all">确定</button>
      </div>
    </div>
  </div>
</template>
