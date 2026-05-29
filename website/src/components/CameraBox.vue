<script setup>
import { Camera, VideoOff } from "lucide-vue-next";
import {
  state,
  startCamera,
  stopCamera,
  videoRef,
  overlayRef,
} from "../composables/useVision";
</script>

<template>
  <article class="card bg-base-100 shadow-xl border border-base-200">
    <div class="card-body p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h2 class="card-title text-xl">Flux Caméra</h2>
          <p class="text-sm text-base-content/60">
            Analyse en temps réel à 3 FPS.
          </p>
        </div>
        <div class="flex gap-2">
          <button
            class="btn btn-primary"
            @click="startCamera"
            :disabled="state.streamOn"
          >
            <Camera class="w-4 h-4" /> Allumer
          </button>
          <button
            class="btn btn-outline btn-error"
            @click="stopCamera"
            :disabled="!state.streamOn"
          >
            <VideoOff class="w-4 h-4" /> Éteindre
          </button>
        </div>
      </div>

      <div
        class="relative w-full aspect-video bg-black rounded-box overflow-hidden shadow-inner"
      >
        <video
          ref="videoRef"
          class="absolute inset-0 h-full w-full object-cover"
          autoplay
          playsinline
          muted
        ></video>
        <canvas
          ref="overlayRef"
          class="pointer-events-none absolute inset-0 h-full w-full"
        ></canvas>

        <div
          v-if="!state.cameraReady"
          class="absolute inset-0 flex items-center justify-center bg-base-300/80 backdrop-blur-sm"
        >
          <span class="text-base-content font-medium"
            >En attente de la caméra...</span
          >
        </div>

        <div
          v-if="state.loading"
          class="absolute top-4 right-4 flex items-center gap-2 bg-black/50 text-white px-3 py-1 rounded-full text-xs backdrop-blur-md"
        >
          <span class="loading loading-ring loading-xs"></span> Analyse
        </div>
      </div>
    </div>
  </article>
</template>
