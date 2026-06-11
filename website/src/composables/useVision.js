import { reactive, ref } from "vue";

// Config API
// Default points at the local backend, proxied by nginx (see website/nginx.conf).
const DEFAULT_ENDPOINT = "/predict";
// Cadence du planificateur de capture. 0 = on réessaie aussi vite que possible
// (le débit réel est borné par le pipeline ci-dessous, pas par ce délai).
const CAPTURE_INTERVAL_MS = 0;
// Largeur max de l'image envoyée. Plus petit = encodage + upload + inférence
// plus rapides (priorité vitesse).
const SEND_WIDTH = 416;
// Profondeur du pipeline : nombre de requêtes en vol simultanées. >1 permet de
// recouvrir la latence réseau d'une frame avec l'inférence de la précédente.
const MAX_IN_FLIGHT = 2;
const API_ENDPOINT = import.meta.env.VITE_ENDPOINT || DEFAULT_ENDPOINT;
const API_KEY = import.meta.env.VITE_API_KEY || "";

// États globaux
export const state = reactive({
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
  fps: 0,
});

// The API key is optional: the dockerized backend is reachable via the nginx
// proxy without one. It is only sent when explicitly configured.
export const configReady = Boolean(API_ENDPOINT);
export const endpointLabel = API_ENDPOINT;
export const apiKeyLabel = API_KEY ? "Configurée" : "Non requise (proxy local)";

// Ref Vue caméra
export const videoRef = ref(null);
export const overlayRef = ref(null);
const captureCanvas = document.createElement("canvas");

let mediaStream = null;
let captureTimer = null;
// Requêtes /predict en cours (pour pouvoir toutes les annuler à l'arrêt).
const inFlightControllers = new Set();
let loopActive = false;
let lastFrameTime = 0;

// --- FONCTIONS ---
function extractDetections(payload) {
  const results = payload?.images?.[0]?.results || [];
  if (!results.length) return [];

  return results.map((item, index) => {
    const label = String(
      item.name || item.class || item.label || `Objet ${index + 1}`,
    );
    const match = label.match(/\d+/);
    const fingerValue = match ? parseInt(match[0], 10) : 0;
    const conf = item.confidence ?? item.score;
    const confidenceText = conf ? `${Math.round(conf * 100)}%` : "";

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

export function redrawOverlay() {
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

  const sentWidth = captureCanvas.width;
  const sentHeight = captureCanvas.height;

  const scaleX = overlay.width / sentWidth;
  const scaleY = overlay.height / sentHeight;

  ctx.lineWidth = 4;
  ctx.font = "bold 16px Space Grotesk, system-ui, sans-serif";
  ctx.textBaseline = "top";

  state.detections.forEach((detection, index) => {
    if (!detection.box) return;
    const { x1, y1, x2, y2 } = detection.box;
    const width = Math.abs(x2 - x1) * scaleX;
    const height = Math.abs(y2 - y1) * scaleY;

    const left = overlay.width - Math.max(x1, x2) * scaleX;
    const top = Math.min(y1, y2) * scaleY;

    const color = ["#2563eb", "#0f766e", "#f97316", "#dc2626", "#8b5cf6"][
      index % 5
    ];
    ctx.strokeStyle = color;
    ctx.strokeRect(left, top, width, height);

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
  if (!video || !video.videoWidth || !video.videoHeight) return;
  if (!configReady) {
    state.error = "Aucun endpoint d'inférence configuré (VITE_ENDPOINT).";
    return;
  }

  state.error = "";
  const controller = new AbortController();
  inFlightControllers.add(controller);
  state.loading = true;

  try {
    const scale = Math.min(1, SEND_WIDTH / video.videoWidth);

    const baseWidth = video.videoWidth * scale;
    const baseHeight = video.videoHeight * scale;

    // L'image envoyée est figée dans des variables locales : plusieurs frames
    // peuvent être en vol en parallèle sans se marcher dessus sur le canvas.
    const sentWidth = Math.round(baseWidth / 32) * 32;
    const sentHeight = Math.round(baseHeight / 32) * 32;
    captureCanvas.width = sentWidth;
    captureCanvas.height = sentHeight;

    const context = captureCanvas.getContext("2d");
    context.drawImage(video, 0, 0, sentWidth, sentHeight);

    const blob = await new Promise((res, rej) =>
      captureCanvas.toBlob(
        (b) => (b ? res(b) : rej(new Error("Erreur capture."))),
        "image/jpeg",
        0.6,
      ),
    );

    const formData = new FormData();
    formData.append("file", blob, `frame.jpg`);

    const response = await fetch(API_ENDPOINT.trim(), {
      method: "POST",
      headers: API_KEY ? { "x-api-key": API_KEY.trim() } : {},
      body: formData,
      signal: controller.signal,
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

    // Live FPS = cadence réelle entre deux frames traitées (lissée).
    const now = performance.now();
    if (lastFrameTime) {
      const instantFps = 1000 / (now - lastFrameTime);
      state.fps = state.fps ? state.fps * 0.8 + instantFps * 0.2 : instantFps;
    }
    lastFrameTime = now;

    redrawOverlay();
  } catch (error) {
    if (error.name !== "AbortError") {
      state.error = `Erreur: ${error.message}`;
      state.status = "Échec de l'analyse.";
    }
  } finally {
    inFlightControllers.delete(controller);
    state.loading = inFlightControllers.size > 0;
  }
}

function captureLoop() {
  if (!loopActive) return;
  // Tant qu'on n'a pas atteint la profondeur de pipeline, on lance une frame
  // sans l'attendre : la suivante se prépare pendant que celle-ci est en vol.
  if (inFlightControllers.size < MAX_IN_FLIGHT) sendFrame();
  const gated = inFlightControllers.size >= MAX_IN_FLIGHT;
  captureTimer = setTimeout(captureLoop, gated ? 5 : CAPTURE_INTERVAL_MS);
}

export function startLoop() {
  stopLoop();
  state.streamOn = true;
  state.status = "Caméra active.";
  loopActive = true;
  lastFrameTime = 0;
  captureLoop();
}

export function stopLoop() {
  state.streamOn = false;
  loopActive = false;
  if (captureTimer) clearTimeout(captureTimer);
  for (const controller of inFlightControllers) controller.abort();
  inFlightControllers.clear();
  captureTimer = null;
  state.loading = false;
  state.fps = 0;
}

export async function startCamera() {
  state.error = "";
  if (!navigator.mediaDevices?.getUserMedia) {
    state.error = "Caméra non supportée.";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        // HD (720p)
        width: { ideal: 1280 },
        height: { ideal: 720 },

        // Full HD (1080p), utilise plutôt ceci :
        // width: { ideal: 1920 },
        // height: { ideal: 1080 }
      },
      audio: false,
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

export function stopCamera() {
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
