<script setup>
import {
  computed,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch,
} from "vue";

const DEFAULT_ENDPOINT = "";
const CAPTURE_INTERVAL_MS = 900;

const API_ENDPOINT = import.meta.env.VITE_ENDPOINT || DEFAULT_ENDPOINT;
const API_KEY = import.meta.env.VITE_API_KEY || "";

const videoRef = ref(null);
const overlayRef = ref(null);
const captureCanvas = document.createElement("canvas");

const state = reactive({
  streamOn: false,
  cameraReady: false,
  loading: false,
  error: "",
  status: "Caméra arrêtée.",
  responseText: "Aucune prédiction pour l’instant.",
  rawResult: null,
  frameAt: "",
  totalFingers: 0,
  detections: [],
});

let mediaStream = null;
let captureTimer = null;
let abortController = null;

const hasVideo = computed(() => state.cameraReady);
const detectionCount = computed(() => state.detections.length);
const endpointLabel = computed(() => API_ENDPOINT);
const apiKeyLabel = computed(() =>
  API_KEY ? "Récupérer avec succès" : "Manquante",
);
const configReady = computed(() => Boolean(API_ENDPOINT && API_KEY));

function formatConfidence(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }

  return `${Math.round(value * 100)}%`;
}

function extractBox(item) {
  const source =
    item?.bbox ??
    item?.box ??
    item?.coordinates ??
    item?.xyxy ??
    item?.points ??
    item?.position;

  if (Array.isArray(source) && source.length >= 4) {
    const [x1, y1, x2, y2] = source.map(Number);
    return { x1, y1, x2, y2 };
  }

  if (source && typeof source === "object") {
    if ("x1" in source && "y1" in source && "x2" in source && "y2" in source) {
      return {
        x1: Number(source.x1),
        y1: Number(source.y1),
        x2: Number(source.x2),
        y2: Number(source.y2),
      };
    }

    if ("x" in source && "y" in source && "w" in source && "h" in source) {
      const x = Number(source.x);
      const y = Number(source.y);
      const w = Number(source.w);
      const h = Number(source.h);
      return { x1: x, y1: y, x2: x + w, y2: y + h };
    }
  }

  return null;
}

function normalizeBox(box) {
  if (!box) return null;

  const values = [box.x1, box.y1, box.x2, box.y2].map(Number);
  if (values.some((value) => Number.isNaN(value))) return null;

  const [x1, y1, x2, y2] = values;
  const maxValue = Math.max(x1, y1, x2, y2);
  const looksNormalized = maxValue <= 1.5;

  return {
    x1: Math.min(x1, x2),
    y1: Math.min(y1, y2),
    x2: Math.max(x1, x2),
    y2: Math.max(y1, y2),
    normalized: looksNormalized,
  };
}

function extractLabel(item, index) {
  return String(
    item?.className ??
      item?.class_name ??
      item?.label ??
      item?.name ??
      item?.class ??
      item?.category ??
      item?.fingerCount ??
      index + 1,
  );
}

function extractDetections(payload) {
  if (!payload || typeof payload !== "object") return [];

  const visited = new Set();

  function collectArrays(node, depth = 0) {
    if (!node || depth > 3) return [];

    if (Array.isArray(node)) {
      return [node];
    }

    if (typeof node !== "object") {
      return [];
    }

    if (visited.has(node)) {
      return [];
    }

    visited.add(node);

    return Object.values(node).flatMap((value) =>
      collectArrays(value, depth + 1),
    );
  }

  const groups = collectArrays(payload).filter((items) =>
    items.some((item) => item && typeof item === "object"),
  );

  const source =
    groups.find((items) =>
      items.some(
        (item) =>
          item?.label ||
          item?.name ||
          item?.class ||
          item?.confidence ||
          item?.score ||
          item?.bbox ||
          item?.box,
      ),
    ) || groups[0];
  if (!source) return [];

  return source.map((item, index) => {
    const box = normalizeBox(extractBox(item));
    const label = extractLabel(item, index);
    const confidence =
      item?.confidence ??
      item?.score ??
      item?.probability ??
      item?.conf ??
      null;
    const fingerValue = Number(
      item?.fingerCount ??
        item?.count ??
        item?.value ??
        item?.class ??
        item?.label,
    );

    return {
      label,
      confidence,
      confidenceText: formatConfidence(confidence),
      box,
      fingerValue: Number.isFinite(fingerValue) ? fingerValue : 1,
    };
  });
}

function redrawOverlay() {
  const video = videoRef.value;
  const overlay = overlayRef.value;

  if (!video || !overlay) return;

  const rect = video.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);

  const ctx = overlay.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!state.detections.length || !video.videoWidth || !video.videoHeight)
    return;

  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  ctx.lineWidth = 4;
  ctx.font = "bold 16px Space Grotesk, system-ui, sans-serif";
  ctx.textBaseline = "top";

  state.detections.forEach((detection, index) => {
    if (!detection.box) return;

    const { x1, y1, x2, y2, normalized } = detection.box;
    const left = normalized
      ? Math.min(x1, x2) * overlay.width
      : Math.min(x1, x2) * scaleX;
    const top = normalized
      ? Math.min(y1, y2) * overlay.height
      : Math.min(y1, y2) * scaleY;
    const width = normalized
      ? Math.abs(x2 - x1) * overlay.width
      : Math.abs(x2 - x1) * scaleX;
    const height = normalized
      ? Math.abs(y2 - y1) * overlay.height
      : Math.abs(y2 - y1) * scaleY;

    const color = ["#2563eb", "#0f766e", "#f97316", "#dc2626", "#8b5cf6"][
      index % 5
    ];

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.strokeRect(left, top, width, height);

    const label = `${detection.label}${detection.confidenceText ? ` ${detection.confidenceText}` : ""}`;
    const labelWidth = Math.max(64, ctx.measureText(label).width + 18);
    const labelHeight = 28;
    const labelY = Math.max(0, top - labelHeight);

    ctx.fillRect(left, labelY, labelWidth, labelHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, left + 8, labelY + 5);
    ctx.fillStyle = color;
  });
}

function computeFingerTotal(detections) {
  return detections.reduce(
    (total, detection) =>
      total +
      (Number.isFinite(detection.fingerValue) ? detection.fingerValue : 1),
    0,
  );
}

async function parsePredictionResponse(response) {
  const rawText = await response.text();

  try {
    return JSON.parse(rawText);
  } catch {
    return rawText;
  }
}

async function sendFrame() {
  const video = videoRef.value;
  if (!video || !video.videoWidth || !video.videoHeight) return;
  if (state.loading) return;

  if (!configReady.value) {
    state.error =
      "Le fichier website/.env doit contenir VITE_ENDPOINT et VITE_API_KEY.";
    return;
  }

  state.loading = true;
  state.error = "";
  state.status = "Analyse de la frame en cours...";

  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  try {
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const context = captureCanvas.getContext("2d");
    if (!context) throw new Error("Impossible de préparer la capture.");

    context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const blob = await new Promise((resolve, reject) => {
      captureCanvas.toBlob(
        (result) => {
          if (!result) {
            reject(new Error("Capture webcam impossible."));
            return;
          }
          resolve(result);
        },
        "image/jpeg",
        0.85,
      );
    });

    const endpoint = API_ENDPOINT.trim().replace(/\/$/, "");
    const formData = new FormData();
    formData.append("file", blob, `frame-${Date.now()}.jpg`);

    const response = await fetch(`${endpoint}/predict`, {
      method: "POST",
      headers: {
        "x-api-key": API_KEY.trim(),
      },
      body: formData,
      signal: abortController.signal,
    });

    const data = await parsePredictionResponse(response);

    if (!response.ok) {
      throw new Error(
        typeof data === "string" ? data : JSON.stringify(data, null, 2),
      );
    }

    state.rawResult = data;
    state.detections = extractDetections(data);
    state.totalFingers = computeFingerTotal(state.detections);
    state.responseText =
      typeof data === "string" ? data : JSON.stringify(data, null, 2);
    state.frameAt = new Date().toLocaleTimeString();
    state.status = `Dernière analyse à ${state.frameAt}.`;

    redrawOverlay();
  } catch (error) {
    if (error.name !== "AbortError") {
      state.error = `Erreur: ${error.message}`;
      state.status = "La requête a échoué.";
    }
  } finally {
    state.loading = false;
  }
}

function startLoop() {
  stopLoop();
  state.streamOn = true;
  state.status = "Caméra active.";
  captureTimer = setInterval(sendFrame, CAPTURE_INTERVAL_MS);
  sendFrame();
}

function stopLoop() {
  state.streamOn = false;
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}

async function startCamera() {
  state.error = "";
  state.cameraReady = false;

  if (!navigator.mediaDevices?.getUserMedia) {
    state.error = "La caméra n’est pas supportée par ce navigateur.";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
      },
      audio: false,
    });

    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream;
      await videoRef.value.play();
    }

    state.cameraReady = true;
    state.status = "Caméra prête.";
    startLoop();
    redrawOverlay();
  } catch (error) {
    state.error = `Impossible d’accéder à la webcam: ${error.message}`;
  }
}

function stopCamera() {
  stopLoop();
  state.cameraReady = false;
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }
  const overlay = overlayRef.value;
  const ctx = overlay?.getContext("2d");
  if (ctx && overlay) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }
  state.status = "Caméra arrêtée.";
}

function resetResult() {
  state.error = "";
  state.responseText = "Aucune prédiction pour l’instant.";
  state.rawResult = null;
  state.totalFingers = 0;
  state.detections = [];
  state.frameAt = "";
  redrawOverlay();
}

watch(
  () => state.detections,
  () => redrawOverlay(),
  { deep: true },
);

onMounted(() => {
  window.addEventListener("resize", redrawOverlay);
});

onBeforeUnmount(() => {
  stopCamera();
  window.removeEventListener("resize", redrawOverlay);
});
</script>

<template>
  <div
    class="min-h-screen bg-gradient-to-br from-slate-50 via-white to-amber-50 text-slate-900"
  >
    <main
      class="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8"
    >
      <header
        class="grid gap-4 rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl md:grid-cols-[1.5fr_0.8fr] md:items-end"
      >
        <div>
          <h1 class="text-4xl font-bold tracking-tight sm:text-5xl">Hand AI</h1>
          <p class="mt-2 text-sm text-slate-500">
            Détection de doigts en temps réel
          </p>
        </div>

        <div
          class="grid gap-3 rounded-3xl border border-slate-200 bg-slate-950 px-5 py-4 text-slate-100 shadow-xl"
        >
          <div class="flex items-center justify-between gap-4">
            <span class="text-sm text-slate-300">Statut</span>
            <span
              class="rounded-full bg-ocean-500/15 px-3 py-1 text-xs font-semibold text-emerald-200"
              >{{ state.streamOn ? "En direct" : "Arrêté" }}</span
            >
          </div>
          <div class="text-2xl font-bold tracking-tight">
            {{ state.totalFingers }} doigts
          </div>
          <p class="text-sm leading-6 text-slate-300">{{ state.status }}</p>
          <p class="text-xs text-slate-400">
            Dernière frame: {{ state.frameAt || "—" }}
          </p>
        </div>
      </header>

      <section class="grid flex-1 gap-6 lg:grid-cols-[1.35fr_0.85fr]">
        <article
          class="rounded-[2rem] border border-white/70 bg-white/80 p-4 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:p-6"
        >
          <div class="mb-4 flex items-center justify-between gap-4">
            <div>
              <h2 class="text-lg font-bold">Flux caméra</h2>
              <p class="text-sm text-slate-500">
                Les boîtes et classes s’affichent au-dessus de la vidéo.
              </p>
            </div>
            <div class="flex gap-3">
              <button
                type="button"
                class="rounded-full bg-ocean-600 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-ocean-500/20 transition hover:bg-ocean-700"
                @click="startCamera"
              >
                Allumer caméra
              </button>
              <button
                type="button"
                class="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                @click="stopCamera"
              >
                Éteindre caméra
              </button>
            </div>
          </div>

          <div
            class="relative overflow-hidden rounded-[1.75rem] border border-slate-200 bg-slate-950/5"
          >
            <div class="relative aspect-[16/10] w-full bg-black">
              <video
                ref="videoRef"
                class="h-full w-full object-cover"
                autoplay
                playsinline
                muted
              ></video>
              <canvas
                ref="overlayRef"
                class="pointer-events-none absolute inset-0 h-full w-full"
              ></canvas>

              <div
                v-if="!hasVideo"
                class="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-950/20 to-slate-950/40 px-6 text-center text-sm leading-6 text-white"
              >
                Lancez la webcam
              </div>
            </div>
          </div>
        </article>

        <aside class="grid gap-6">
          <article
            class="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Configuration chargée</h2>
            <p class="mt-1 text-sm text-slate-500">
              Endpoint chargé depuis le .env du dossier website.
            </p>

            <div class="mt-5 grid gap-4 text-sm">
              <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p
                  class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500"
                >
                  Endpoint
                </p>
                <p class="mt-2 break-all font-medium text-slate-900">
                  {{ endpointLabel }}
                </p>
              </div>

              <div class="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                <p
                  class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500"
                >
                  Clé API
                </p>
                <p class="mt-2 font-medium text-slate-900">{{ apiKeyLabel }}</p>
              </div>

              <button
                type="button"
                class="w-full rounded-full bg-gradient-to-r from-ocean-600 to-ocean-700 px-5 py-3 text-sm font-bold text-white shadow-lg shadow-ocean-500/20 transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-50"
                @click="sendFrame"
                :disabled="!state.streamOn || state.loading"
              >
                {{ state.loading ? "Analyse en cours..." : "Analyser" }}
              </button>

              <p
                v-if="state.error"
                class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700"
              >
                {{ state.error }}
              </p>
            </div>
          </article>

          <article
            class="rounded-[2rem] border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Détections</h2>
            <p class="mt-1 text-sm text-slate-500">
              Les classes numériques sont additionnées pour obtenir le total de
              doigts.
            </p>

            <div class="mt-5 grid gap-3">
              <div
                v-for="(detection, index) in state.detections"
                :key="index"
                class="rounded-2xl border border-slate-200 bg-slate-50 p-4"
              >
                <div class="flex items-center justify-between gap-4">
                  <div>
                    <p class="text-sm font-bold text-slate-900">
                      {{ detection.label }}
                    </p>
                    <p class="mt-1 text-xs text-slate-500">
                      Box {{ index + 1 }}
                    </p>
                  </div>
                  <span
                    class="rounded-full bg-ocean-500/10 px-3 py-1 text-xs font-semibold text-ocean-700"
                    >{{ detection.confidenceText || "n/a" }}</span
                  >
                </div>
                <p
                  v-if="detection.box"
                  class="mt-3 break-words text-xs leading-5 text-slate-600"
                >
                  Boîte: {{ JSON.stringify(detection.box) }}
                </p>
              </div>
              <div
                v-if="!state.detections.length"
                class="rounded-2xl border border-dashed border-slate-300 bg-sand-50 px-4 py-5 text-sm leading-6 text-slate-500"
              >
                Lance la webcam puis clique sur “Analyser maintenant” pour
                tester la chaîne API.
              </div>
            </div>
          </article>

          <article
            class="rounded-4xl border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Réponse brute</h2>
            <pre
              class="mt-4 max-h-72 overflow-auto rounded-3xl bg-slate-950 p-5 text-xs leading-6 text-slate-100"
              >{{ state.responseText }}</pre
            >
          </article>
        </aside>
      </section>

      <footer class="pb-2 text-center text-xs text-slate-500">
        Hand AI Dashboard
      </footer>
    </main>
  </div>
</template>
