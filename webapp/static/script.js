/* ──────────────────────────────────────────────────────────────────────────
   Clasificador de Dieta Animal — lógica del cliente
   Subida + preview → animación de "análisis" (rasgos) → resultado en grande.
   ────────────────────────────────────────────────────────────────────────── */

"use strict";

// Etiquetas de presentación por clase del modelo.
const DISPLAY = {
  carnivore: { name: "Carnívoro",   emoji: "🦁", cls: "diet-carnivore" },
  herbivore: { name: "Herbívoro",   emoji: "🦌", cls: "diet-herbivore" },
  omnivore:  { name: "Omnívoro",    emoji: "🐻", cls: "diet-omnivore"  },
  unknown:   { name: "Desconocido", emoji: "❓", cls: "diet-unknown"   },
};

// Pasos del "análisis": cosmético, pero evoca los rasgos morfológicos reales.
const ANALYZING_STEPS = [
  "Cargando imagen…",
  "Detectando al animal…",
  "Analizando posición de los ojos…",
  "Examinando dentadura y mandíbula…",
  "Evaluando garras y extremidades…",
  "Midiendo proporciones del cuerpo…",
  "Extrayendo características con ResNet…",
  "Comparando con la base morfológica…",
  "Calculando probabilidades de dieta…",
];

const MIN_ANIM_MS = 2600;   // duración mínima de la animación, aunque el server sea instantáneo
const STEP_MS     = 480;    // cada cuánto cambia el mensaje

// ── Referencias al DOM ──────────────────────────────────────────────────────
const fileInput   = document.getElementById("file-input");
const dropZone    = document.getElementById("drop-zone");
const placeholder = document.getElementById("placeholder");
const previewWrap = document.getElementById("preview-wrap");
const previewImg  = document.getElementById("preview-img");
const statusEl    = document.getElementById("status");
const resultEl    = document.getElementById("result");
const analyzeBtn  = document.getElementById("analyze-btn");
const resetBtn    = document.getElementById("reset-btn");

let currentFile = null;
let stepTimer   = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ── Selección de archivo ────────────────────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) handleFile(file);
});

// Arrastrar y soltar
["dragenter", "dragover"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  dropZone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

// Pegar desde el portapapeles (Ctrl+V) en cualquier parte de la página.
document.addEventListener("paste", (e) => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) {
        e.preventDefault();
        handleFile(file);
      }
      break;
    }
  }
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    placeholder.hidden = true;
    previewWrap.hidden = false;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    // Limpia cualquier resultado anterior.
    resultEl.className = "result";
    resultEl.innerHTML = "";
  };
  reader.readAsDataURL(file);
}

// ── Animación ───────────────────────────────────────────────────────────────
function startAnimation() {
  previewWrap.classList.add("analyzing");
  resultEl.className = "result";          // oculta resultado previo
  let i = 0;
  statusEl.textContent = ANALYZING_STEPS[0];
  statusEl.classList.add("show");
  stepTimer = setInterval(() => {
    i = (i + 1) % ANALYZING_STEPS.length;
    statusEl.textContent = ANALYZING_STEPS[i];
  }, STEP_MS);
}

function stopAnimation() {
  clearInterval(stepTimer);
  stepTimer = null;
  previewWrap.classList.remove("analyzing");
  statusEl.classList.remove("show");
}

// ── Analizar ────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  startAnimation();

  const t0 = performance.now();
  const form = new FormData();
  form.append("image", currentFile);

  let data;
  try {
    const resp = await fetch("/api/predict", { method: "POST", body: form });
    data = await resp.json();
  } catch (err) {
    data = { ok: false, error: "No se pudo contactar con el servidor." };
  }

  // Deja que la animación se vea un mínimo de tiempo.
  const elapsed = performance.now() - t0;
  if (elapsed < MIN_ANIM_MS) await sleep(MIN_ANIM_MS - elapsed);

  stopAnimation();
  reveal(data);

  analyzeBtn.disabled = false;
  resetBtn.disabled = false;
}

// ── Mostrar resultado ───────────────────────────────────────────────────────
function reveal(data) {
  if (!data || !data.ok) {
    const msg = (data && data.error) || "Error desconocido.";
    resultEl.className = "result error-result show";
    resultEl.innerHTML = `<div class="err">⚠️ ${msg}</div>`;
    return;
  }

  const d = DISPLAY[data.label] || DISPLAY.unknown;
  const pct = (data.confidence * 100).toFixed(1);
  const model =
    data.source === "resnet50" ? "ResNet-50" :
    data.source === "resnet18" ? "ResNet-18" : data.source;

  resultEl.className = `result show ${d.cls}`;
  resultEl.innerHTML = `
    <div class="result-emoji">${d.emoji}</div>
    <div class="result-name">${d.name}</div>
    <div class="conf">
      <div class="conf-bar"><div class="conf-fill"></div></div>
      <div class="conf-label">${pct}% de confianza</div>
    </div>
    <div class="model-badge">Resuelto por ${model}</div>
  `;

  // Anima el llenado de la barra tras el primer frame.
  requestAnimationFrame(() => {
    const fill = resultEl.querySelector(".conf-fill");
    if (fill) fill.style.width = pct + "%";
  });
}

// ── Reiniciar ───────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", reset);

function reset() {
  currentFile = null;
  fileInput.value = "";
  previewImg.src = "";
  previewWrap.hidden = true;
  placeholder.hidden = false;
  stopAnimation();
  resultEl.className = "result";
  resultEl.innerHTML = "";
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
}
