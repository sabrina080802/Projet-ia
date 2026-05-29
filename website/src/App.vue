<script setup>
import {
  computed,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch,
} from "vue";

const DEFAULT_ENDPOINT = ""; // laisser vide, on utilise le .env et les variables GitHub pour Pages
const CAPTURE_INTERVAL_MS = 300; // environ 3-4 fps

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
const endpointLabel = computed(() => API_ENDPOINT);
const apiKeyLabel = computed(() =>
  API_KEY ? "Récupérée avec succès" : "Manquante",
);
const configReady = computed(() => Boolean(API_ENDPOINT && API_KEY));

function extractDetections(payload) {
  const results = payload?.images?.[0]?.results || [];
  if (!results.length) return [];

  return results.map((item, index) => {
    // Label (ex: "hand_5", "class_3", etc.)
    const label = String(
      item.name || item.class || item.label || `Objet ${index + 1}`,
    );

    // Compteur des doigts
    const match = label.match(/\d+/);
    const fingerValue = match ? parseInt(match[0], 10) : 0;

    // Niveau de confiance en %
    const conf = item.confidence ?? item.score;
    const confidenceText = conf ? `${Math.round(conf * 100)}%` : "";

    // Bounding Box (me demander pas d'expliquer, c('est un calcul qui fonctionne)
    let box = null;
    const b = item.box || item.bbox || item;
    if (b && typeof b === "object") {
      if ("x1" in b && "y1" in b && "x2" in b && "y2" in b) {
        box = { x1: b.x1, y1: b.y1, x2: b.x2, y2: b.y2 };
      } else if ("x" in b && "y" in b && ("w" in b || "width" in b)) {
        const w = b.w ?? b.width;
        const h = b.h ?? b.height;
        box = { x1: b.x, y1: b.y, x2: b.x + w, y2: b.y + h };
      }
    }

    return { label, confidenceText, fingerValue, box };
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

    const { x1, y1, x2, y2 } = detection.box;
    const left = Math.min(x1, x2) * scaleX;
    const top = Math.min(y1, y2) * scaleY;
    const width = Math.abs(x2 - x1) * scaleX;
    const height = Math.abs(y2 - y1) * scaleY;

    // Couleur pour les boîtes
    const color = ["#2563eb", "#0f766e", "#f97316", "#dc2626", "#8b5cf6"][
      index % 5
    ];

    // Dessin de la boîte
    ctx.strokeStyle = color;
    ctx.strokeRect(left, top, width, height);

    // Dessin du texte
    const label = `${detection.label}${detection.confidenceText ? ` ${detection.confidenceText}` : ""}`;
    const labelWidth = Math.max(64, ctx.measureText(label).width + 18);
    ctx.fillStyle = color;
    ctx.fillRect(left, Math.max(0, top - 28), labelWidth, 28);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, left + 8, Math.max(0, top - 28) + 5);
  });
}

async function sendFrame() {
  const video = videoRef.value;
  if (!video || !video.videoWidth || !video.videoHeight || state.loading)
    return;

  if (!configReady.value) {
    state.error =
      "Le fichier .env doit contenir VITE_ENDPOINT et VITE_API_KEY.";
    return;
  }

  state.loading = true;
  state.error = "";

  if (abortController) abortController.abort();
  abortController = new AbortController();

  try {
    const MAX_WIDTH = 640; // ne pas dépasser 640px de large (optimisation du flux envoyé à l'API)
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);

    captureCanvas.width = video.videoWidth * scale;
    captureCanvas.height = video.videoHeight * scale;
    const context = captureCanvas.getContext("2d");

    // redimensionnement pour l'API
    context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const blob = await new Promise((resolve, reject) => {
      // compression
      captureCanvas.toBlob(
        (res) => (res ? resolve(res) : reject(new Error("Erreur de capture."))),
        "image/jpeg",
        0.6, // 60% -> évite d'envoyer images trop lourdes (j'ai une caméra 4k)
      );
    });

    const endpoint = API_ENDPOINT.trim().replace(/\/$/, "");
    const formData = new FormData();
    formData.append("file", blob, `frame.jpg`);

    const response = await fetch(`${endpoint}/predict`, {
      method: "POST",
      headers: { "x-api-key": API_KEY.trim() },
      body: formData,
      signal: abortController.signal,
    });

    const rawText = await response.text();
    const data = rawText.startsWith("{") ? JSON.parse(rawText) : rawText;

    if (!response.ok)
      throw new Error(typeof data === "string" ? data : JSON.stringify(data));

    state.rawResult = data;
    state.detections = extractDetections(data);
    state.totalFingers = state.detections.reduce(
      (sum, det) => sum + det.fingerValue,
      0,
    );
    state.responseText =
      typeof data === "string" ? data : JSON.stringify(data, null, 2);
    state.frameAt = new Date().toLocaleTimeString();
    state.status = `Analyse réussie.`;

    redrawOverlay();
  } catch (error) {
    if (error.name !== "AbortError") {
      state.error = `Erreur: ${error.message}`;
      state.status = "Échec de l'analyse.";
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
  if (captureTimer) clearInterval(captureTimer);
  if (abortController) abortController.abort();
  captureTimer = null;
  abortController = null;
}

async function startCamera() {
  state.error = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    state.error = "Caméra non supportée.";
    return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });
    if (videoRef.value) {
      videoRef.value.srcObject = mediaStream;
      await videoRef.value.play();
    }
    state.cameraReady = true;
    startLoop();
  } catch (error) {
    state.error = `Erreur webcam: ${error.message}`;
  }
}

function stopCamera() {
  stopLoop();
  state.cameraReady = false;
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (videoRef.value) videoRef.value.srcObject = null;
  const ctx = overlayRef.value?.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, overlayRef.value.width, overlayRef.value.height);
  state.status = "Caméra arrêtée.";
  state.detections = [];
  state.totalFingers = 0;
}

watch(() => state.detections, redrawOverlay, { deep: true });

onMounted(() => window.addEventListener("resize", redrawOverlay));
onBeforeUnmount(() => {
  stopCamera();
  window.removeEventListener("resize", redrawOverlay);
});
</script>

<template>
  <div
    class="min-h-screen bg-linear-to-br from-slate-50 via-white to-amber-50 text-slate-900"
  >
    <main
      class="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8"
    >
      <header
        class="grid gap-4 rounded-4xl border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl md:grid-cols-[1.5fr_0.8fr] md:items-end"
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
              class="rounded-full bg-blue-500/15 px-3 py-1 text-xs font-semibold text-emerald-200"
            >
              {{ state.streamOn ? "En direct" : "Arrêté" }}
            </span>
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
          class="rounded-4xl border border-white/70 bg-white/80 p-4 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl sm:p-6"
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
                class="rounded-full bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:bg-blue-700"
                @click="startCamera"
              >
                Allumer
              </button>
              <button
                type="button"
                class="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                @click="stopCamera"
              >
                Éteindre
              </button>
            </div>
          </div>

          <div
            class="relative overflow-hidden rounded-[1.75rem] border border-slate-200 bg-slate-950/5"
          >
            <div
              class="relative w-full aspect-video bg-black rounded-3xl overflow-hidden shadow-inner"
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
                v-if="!hasVideo"
                class="absolute inset-0 flex items-center justify-center bg-slate-900/80 px-6 text-center text-sm font-medium text-white backdrop-blur-sm"
              >
                Lancez la webcam pour commencer la détection
              </div>
            </div>
          </div>
        </article>

        <aside class="grid gap-6 max-h-[calc(100vh-8rem)] overflow-y-auto pr-2">
          <article
            class="rounded-4xl border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Configuration chargée</h2>
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

              <p
                v-if="state.error"
                class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700"
              >
                {{ state.error }}
              </p>
            </div>
          </article>

          <article
            class="rounded-4xl border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Détections actuelles</h2>
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
                      Doigts: {{ detection.fingerValue }}
                    </p>
                  </div>
                  <span
                    class="rounded-full bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-700"
                  >
                    {{ detection.confidenceText || "n/a" }}
                  </span>
                </div>
              </div>
              <div
                v-if="!state.detections.length"
                class="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-5 text-sm text-slate-500"
              >
                Aucune détection pour le moment.
              </div>
            </div>
          </article>

          <article
            class="rounded-4xl border border-white/70 bg-white/80 p-6 shadow-[0_24px_70px_rgba(15,23,42,0.08)] backdrop-blur-xl"
          >
            <h2 class="text-lg font-bold">Réponse brute API</h2>
            <pre
              class="mt-4 max-h-72 overflow-auto rounded-3xl bg-slate-950 p-5 text-xs leading-6 text-slate-100"
              >{{ state.responseText }}</pre
            >
          </article>
        </aside>
      </section>
    </main>
  </div>
</template>
