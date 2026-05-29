<script setup>
import { Activity, Settings, Code, Fingerprint } from "lucide-vue-next";
import { state, endpointLabel, apiKeyLabel } from "../composables/useVision";
</script>

<template>
  <div class="flex flex-col gap-4 pb-4">
    <div
      class="bg-primary text-primary-content rounded-box p-6 shadow-lg text-center relative overflow-hidden"
    >
      <Fingerprint class="absolute -right-6 -bottom-6 w-32 h-32 opacity-10" />

      <p
        class="text-sm font-bold uppercase tracking-widest opacity-80 relative z-10"
      >
        Total Doigts Levés
      </p>
      <p class="text-[5.5rem] leading-none font-black mt-2 relative z-10">
        {{ state.totalFingers }}
      </p>
    </div>

    <div class="card bg-base-100 shadow-sm border border-base-200">
      <div class="card-body p-5">
        <h2
          class="font-bold flex items-center gap-2 text-sm uppercase tracking-wide text-base-content/70"
        >
          <Settings class="w-4 h-4" /> API
        </h2>
        <div class="mt-2 space-y-2">
          <div
            class="bg-base-200 rounded-lg p-2.5 flex justify-between items-center"
          >
            <span class="text-xs font-bold opacity-60">Endpoint</span>
            <span class="text-xs font-mono max-w-[12rem] truncate">{{
              endpointLabel || "Aucun"
            }}</span>
          </div>
          <div
            class="bg-base-200 rounded-lg p-2.5 flex justify-between items-center"
          >
            <span class="text-xs font-bold opacity-60">Clé API</span>
            <span class="text-xs">{{ apiKeyLabel }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="card bg-base-100 shadow-sm border border-base-200">
      <div class="card-body p-5">
        <h2
          class="font-bold flex items-center gap-2 text-sm uppercase tracking-wide text-base-content/70"
        >
          <Activity class="w-4 h-4" /> Cibles Détectées
        </h2>
        <div class="flex flex-col gap-2 mt-2">
          <div
            v-for="(detection, index) in state.detections"
            :key="index"
            class="bg-base-200 rounded-lg p-3 flex justify-between items-center"
          >
            <div>
              <p class="font-bold text-sm">{{ detection.label }}</p>
              <p class="text-xs opacity-60 mt-0.5">
                Valeur: {{ detection.fingerValue }}
              </p>
            </div>
            <div class="badge badge-primary badge-outline text-xs">
              {{ detection.confidenceText }}
            </div>
          </div>
          <div
            v-if="!state.detections.length"
            class="text-center py-6 text-sm opacity-50 border border-dashed border-base-300 rounded-lg"
          >
            Aucun doigt visible
          </div>
        </div>
      </div>
    </div>

    <div class="card bg-base-100 shadow-sm border border-base-200">
      <div class="card-body p-0 overflow-hidden">
        <div class="p-4 border-b border-base-200 bg-base-200/50">
          <h2
            class="font-bold flex items-center gap-2 text-sm uppercase tracking-wide text-base-content/70"
          >
            <Code class="w-4 h-4" /> Payload Brut
          </h2>
        </div>
        <div
          class="bg-neutral text-neutral-content text-xs p-4 overflow-auto max-h-48 custom-scrollbar"
        >
          <pre><code>{{ state.responseText }}</code></pre>
        </div>
      </div>
    </div>
  </div>
</template>
