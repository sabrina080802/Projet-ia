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
  <div class="min-h-screen bg-base-200 text-base-content pb-10">
    <main class="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6">
      <header
        class="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-base-100 p-6 rounded-box shadow-xl border border-base-200"
      >
        <div>
          <h1
            class="text-4xl font-black bg-linear-to-r from-primary to-secondary bg-clip-text text-transparent"
          >
            Projet SHARP
          </h1>
          <p class="text-sm font-medium mt-1 opacity-60">
            Détection de doigts levés en temps réel
          </p>
        </div>

        <div class="stats bg-neutral text-neutral-content shadow">
          <div class="stat py-2">
            <div class="stat-title text-neutral-content/70">Total Doigts</div>
            <div class="stat-value text-primary">{{ state.totalFingers }}</div>
          </div>
          <div class="stat py-2">
            <div class="stat-title text-neutral-content/70">Statut IA</div>
            <div class="text-sm font-bold mt-1 flex items-center gap-2">
              <span
                class="w-2 h-2 rounded-full"
                :class="
                  state.streamOn ? 'bg-success animate-pulse' : 'bg-error'
                "
              ></span>
              {{ state.streamOn ? "En direct" : "Arrêté" }}
            </div>
            <div class="stat-desc text-neutral-content/50">
              {{ state.frameAt || "—" }}
            </div>
          </div>
        </div>
      </header>

      <section class="grid flex-1 gap-6 lg:grid-cols-[1.5fr_1fr]">
        <CameraBox />
        <DataSidebar />
      </section>
    </main>
  </div>
</template>
