<script setup>
import { Activity, Settings, Code, Fingerprint } from "lucide-vue-next";
import { state, endpointLabel, apiKeyLabel } from "../composables/useVision";
</script>

<template>
  <aside class="grid gap-6 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2">
    <div class="card bg-base-100 shadow-xl border border-base-200">
      <div class="card-body p-6">
        <h2 class="card-title text-lg">
          <Settings class="w-5 h-5 text-primary" /> Configuration
        </h2>
        <div class="bg-base-200 rounded-box p-3 mt-2">
          <p class="text-xs font-bold text-base-content/50 uppercase">
            Endpoint
          </p>
          <p class="text-sm truncate">{{ endpointLabel }}</p>
        </div>
        <div class="bg-base-200 rounded-box p-3">
          <p class="text-xs font-bold text-base-content/50 uppercase">
            Clé API
          </p>
          <p class="text-sm">{{ apiKeyLabel }}</p>
        </div>
        <div
          v-if="state.error"
          class="alert alert-error mt-2 shadow-sm text-sm"
        >
          {{ state.error }}
        </div>
      </div>
    </div>

    <div class="card bg-base-100 shadow-xl border border-base-200">
      <div class="card-body p-6">
        <h2 class="card-title text-lg">
          <Activity class="w-5 h-5 text-primary" /> Détections
        </h2>
        <div class="flex flex-col gap-2 mt-2">
          <div
            v-for="(detection, index) in state.detections"
            :key="index"
            class="bg-base-200 rounded-box p-4 flex justify-between items-center"
          >
            <div>
              <p class="font-bold text-sm">{{ detection.label }}</p>
              <p
                class="text-xs text-base-content/60 flex items-center gap-1 mt-1"
              >
                <Fingerprint class="w-3 h-3" /> Doigts estimés:
                {{ detection.fingerValue }}
              </p>
            </div>
            <div class="badge badge-primary badge-outline">
              {{ detection.confidenceText }}
            </div>
          </div>
          <div
            v-if="!state.detections.length"
            class="text-center p-4 text-sm text-base-content/50 border border-dashed border-base-300 rounded-box"
          >
            Aucun objet détecté
          </div>
        </div>
      </div>
    </div>

    <div class="card bg-base-100 shadow-xl border border-base-200">
      <div class="card-body p-6">
        <h2 class="card-title text-lg">
          <Code class="w-5 h-5 text-primary" /> Réponse API
        </h2>
        <div class="mockup-code mt-2 text-xs bg-neutral text-neutral-content">
          <pre
            class="overflow-auto max-h-48 px-4"
          ><code>{{ state.responseText }}</code></pre>
        </div>
      </div>
    </div>
  </aside>
</template>
