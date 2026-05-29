<script setup>
import { onMounted, onBeforeUnmount, watch } from "vue";
import { state, redrawOverlay, stopCamera } from "./composables/useVision";
import CameraBox from "./components/CameraBox.vue";
import DataSidebar from "./components/DataSidebar.vue";

watch(() => state.detections, redrawOverlay, { deep: true });
onMounted(() => window.addEventListener("resize", redrawOverlay));
onBeforeUnmount(() => {
  stopCamera();
  window.removeEventListener("resize", redrawOverlay);
});
</script>

<template>
  <div
    class="h-screen w-full bg-base-200 text-base-content flex flex-col p-4 gap-4 overflow-hidden"
  >
    <header
      class="shrink-0 flex justify-between items-center bg-base-100 px-6 py-4 rounded-box shadow-sm border border-base-200"
    >
      <div>
        <h1
          class="text-3xl font-black bg-linear-to-r from-primary to-secondary bg-clip-text text-transparent tracking-tight"
        >
          Projet SHARP
        </h1>
        <p class="text-sm font-medium mt-1 opacity-60">
          Il compte le nombre de doigts levés.
        </p>
      </div>

      <div class="flex items-center gap-4">
        <div class="text-right hidden sm:block">
          <div class="text-xs uppercase tracking-wider font-bold opacity-50">
            Statut du flux
          </div>
          <div
            class="text-sm font-bold flex items-center justify-end gap-2 mt-0.5"
          >
            <span
              class="w-2 h-2 rounded-full"
              :class="state.streamOn ? 'bg-success animate-pulse' : 'bg-error'"
            ></span>
            {{ state.streamOn ? "En direct" : "Arrêté" }}
          </div>
        </div>
      </div>
    </header>

    <main class="flex-1 min-h-0 flex gap-4 max-w-480 mx-auto w-full">
      <section class="flex-1 min-w-0 h-full">
        <CameraBox />
      </section>

      <aside
        class="w-80 lg:w-[24rem] shrink-0 h-full overflow-y-auto pr-2 custom-scrollbar"
      >
        <DataSidebar />
      </aside>
    </main>
  </div>
</template>

<style>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background-color: oklch(var(--bc) / 0.2);
  border-radius: 10px;
}
</style>
