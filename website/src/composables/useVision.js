import { reactive, ref } from "vue";

// Config API
const DEFAULT_ENDPOINT = "";
const CAPTURE_INTERVAL_MS = 300;
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
});

export const configReady = Boolean(API_ENDPOINT && API_KEY);
export const endpointLabel = API_ENDPOINT;
export const apiKeyLabel = API_KEY ? "Récupérée avec succès" : "Manquante";

// Ref Vue caméra
export const videoRef = ref(null);
export const overlayRef = ref(null);
const captureCanvas = document.createElement("canvas");

let mediaStream = null;
let captureTimer = null;
let abortController = null;

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

  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  ctx.lineWidth = 4;
  ctx.font = "bold 16px Space Grotesk, system-ui, sans-serif";
  ctx.textBaseline = "top";

  state.detections.forEach((detection, index) => {
    if (!detection.box) return;
    const { x1, y1, x2, y2 } = detection.box;
    const width = Math.abs(x2 - x1) * scaleX;
    const height = Math.abs(y2 - y1) * scaleY;

    // vidéo en miroir donc on inverse l'axe X /!\
    const left = overlay.width - Math.min(x1, x2) * scaleX - width;
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
  if (!video || !video.videoWidth || !video.videoHeight || state.loading)
    return;
  if (!configReady) {
    state.error =
      "Le fichier .env doit contenir VITE_ENDPOINT et VITE_API_KEY.";
    return;
  }

  state.loading = true;
  state.error = "";
  if (abortController) abortController.abort();
  abortController = new AbortController();

  try {
    const scale = Math.min(1, 640 / video.videoWidth);
    captureCanvas.width = video.videoWidth * scale;
    captureCanvas.height = video.videoHeight * scale;
    const context = captureCanvas.getContext("2d");
    context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    const blob = await new Promise((res, rej) =>
      captureCanvas.toBlob(
        (b) => (b ? res(b) : rej(new Error("Erreur capture."))),
        "image/jpeg",
        0.6,
      ),
    );

    const formData = new FormData();
    formData.append("file", blob, `frame.jpg`);

    const response = await fetch(`${API_ENDPOINT.replace(/\/$/, "")}/predict`, {
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

export function startLoop() {
  stopLoop();
  state.streamOn = true;
  state.status = "Caméra active.";
  captureTimer = setInterval(sendFrame, CAPTURE_INTERVAL_MS);
  sendFrame();
}

export function stopLoop() {
  state.streamOn = false;
  if (captureTimer) clearInterval(captureTimer);
  if (abortController) abortController.abort();
  captureTimer = null;
  abortController = null;
}

export async function startCamera() {
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
