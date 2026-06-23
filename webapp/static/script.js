/* ──────────────────────────────────────────────────────────────────────────
   Clasificador de Dieta Animal — lógica del cliente ("Ficha de naturalista")
   Subida + montura → examen con lupa → SELLO de determinación + regla de confianza.
   Respeta el contrato del backend: POST /api/predict (multipart 'image')
     éxito → { ok:true, label, confidence, source }
     error → { ok:false, error }   (status 400/503)
   label ∈ {carnivore, herbivore, omnivore, other, unknown}
   source ∈ {resnet18, resnet50, morphology, no_image}
   ────────────────────────────────────────────────────────────────────────── */

"use strict";

// Nombre de presentación y clase de color por dieta.
const DISPLAY = {
  carnivore: { name: "Carnívoro",   cls: "diet-carnivore" },
  herbivore: { name: "Herbívoro",   cls: "diet-herbivore" },
  omnivore:  { name: "Omnívoro",    cls: "diet-omnivore"  },
  unknown:   { name: "Indeterminado", cls: "diet-unknown" },
};

// Pasos del examen: cosméticos, evocan los rasgos morfológicos reales.
const EXAM_STEPS = [
  "Desplegando el ejemplar sobre la mesa…",
  "Observando la posición de los ojos…",
  "Examinando la dentadura y la mandíbula…",
  "Anotando garras y extremidades…",
  "Midiendo las proporciones del cuerpo…",
  "Extrayendo rasgos con la red neuronal…",
  "Cotejando con la colección de referencia…",
  "Redactando la determinación…",
];

const MIN_ANIM_MS = 2600;   // duración mínima del examen, aunque el server sea instantáneo
const STEP_MS     = 520;    // cada cuánto cambia la anotación

// ── Referencias al DOM ──────────────────────────────────────────────────────
const fileInput   = document.getElementById("file-input");
const mount       = document.getElementById("mount");
const placeholder = document.getElementById("mount-placeholder");
const specimen    = document.getElementById("specimen");
const specimenImg = document.getElementById("specimen-img");
const statusEl    = document.getElementById("scan-status");
const verdictEl   = document.getElementById("verdict");
const analyzeBtn  = document.getElementById("analyze-btn");
const resetBtn    = document.getElementById("reset-btn");
const specimenNo  = document.getElementById("specimen-no");
const fieldDate   = document.getElementById("field-date");
const fieldDet    = document.getElementById("field-det");

let currentFile = null;
let stepTimer   = null;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

const rnd = (a, b) => Math.floor(Math.random() * (b - a + 1)) + a;

// ── Sellos de catálogo: nº de espécimen aleatorio + fecha de hoy ────────────
specimenNo.textContent = `N.º ${rnd(10, 99)}-${String(rnd(1, 999)).padStart(3, "0")}`;
fieldDate.textContent  = new Date().toLocaleDateString("es-ES",
  { day: "numeric", month: "long", year: "numeric" });

// ── Selección de archivo ────────────────────────────────────────────────────
fileInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) handleFile(file);
});

// Arrastrar y soltar
["dragenter", "dragover"].forEach((ev) =>
  mount.addEventListener(ev, (e) => { e.preventDefault(); mount.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  mount.addEventListener(ev, (e) => { e.preventDefault(); mount.classList.remove("dragover"); })
);
mount.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) handleFile(file);
});

// Pegar desde el portapapeles (Ctrl+V)
document.addEventListener("paste", (e) => {
  const items = (e.clipboardData && e.clipboardData.items) || [];
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      const file = item.getAsFile();
      if (file) { e.preventDefault(); handleFile(file); }
      break;
    }
  }
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    specimenImg.src = e.target.result;
    placeholder.hidden = true;
    specimen.hidden = false;
    analyzeBtn.disabled = false;
    resetBtn.disabled = false;
    clearVerdict();
  };
  reader.readAsDataURL(file);
}

// ── Examen (animación) ──────────────────────────────────────────────────────
function startExam() {
  specimen.classList.add("analyzing");
  clearVerdict();
  let i = 0;
  statusEl.textContent = EXAM_STEPS[0];
  statusEl.classList.add("show");
  stepTimer = setInterval(() => {
    i = (i + 1) % EXAM_STEPS.length;
    statusEl.textContent = EXAM_STEPS[i];
  }, STEP_MS);
}

function stopExam() {
  clearInterval(stepTimer);
  stepTimer = null;
  specimen.classList.remove("analyzing");
  statusEl.classList.remove("show");
  statusEl.textContent = "";
}

function clearVerdict() {
  verdictEl.className = "verdict";
  verdictEl.innerHTML = "";
  if (fieldDet) fieldDet.textContent = "";
}

// ── Analizar ────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", analyze);

async function analyze() {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
  startExam();

  const t0 = performance.now();
  const form = new FormData();
  form.append("image", currentFile);

  let data;
  try {
    const resp = await fetch("/api/predict", { method: "POST", body: form });
    data = await resp.json();
  } catch (err) {
    data = { ok: false, error: "No se pudo contactar con el gabinete (servidor)." };
  }

  // El examen se ve un mínimo de tiempo aunque el server responda al instante.
  const elapsed = performance.now() - t0;
  if (elapsed < MIN_ANIM_MS) await sleep(MIN_ANIM_MS - elapsed);

  stopExam();
  reveal(data);

  analyzeBtn.disabled = false;
  resetBtn.disabled = false;
}

// ── Mostrar la determinación ────────────────────────────────────────────────
function reveal(data) {
  // Error del backend → "nota del conservador" en papel.
  if (!data || !data.ok) {
    const msg = (data && data.error) || "Error desconocido.";
    verdictEl.className = "verdict show";
    verdictEl.innerHTML = `
      <div class="curator-note">
        <span class="note-head">Nota del conservador</span>
        <p>${escapeHtml(msg)}</p>
      </div>`;
    return;
  }

  // "other" = fuera de alcance: no es un animal clasificable por dieta.
  if (data.label === "other") {
    verdictEl.className = "verdict show";
    verdictEl.innerHTML = `
      <div class="stamp diet-other stamp-anim">
        <span class="stamp-kicker">— Fuera de alcance —</span>
        <span class="stamp-name">Fuera de catálogo</span>
      </div>
      <p class="verdict-note">Este ejemplar no corresponde a un animal
        clasificable por dieta.</p>`;
    if (fieldDet) fieldDet.textContent = "fuera de catálogo";
    return;
  }

  const d   = DISPLAY[data.label] || DISPLAY.unknown;
  const pct = Math.max(0, Math.min(100, data.confidence * 100));
  const pctTxt = pct.toFixed(1).replace(".", ",");
  const morph = data.source === "morphology"
    ? `<p class="verdict-note">Determinado por examen de las características físicas
         del ejemplar — dentadura, ojos y extremidades.</p>`
    : "";

  verdictEl.className = "verdict show";
  verdictEl.innerHTML = `
    <div class="stamp ${d.cls} stamp-anim">
      <span class="stamp-kicker">— Dieta determinada —</span>
      <span class="stamp-name">${d.name}</span>
    </div>
    <div class="gauge ${d.cls}">
      <div class="gauge-track">
        <span class="gauge-ticks"></span>
        <span class="gauge-needle"></span>
      </div>
      <div class="gauge-scaleline"><span>0</span><span>50</span><span>100</span></div>
      <div class="gauge-read">Confianza de la determinación: <b>${pctTxt} %</b></div>
    </div>
    ${morph}`;

  if (fieldDet) fieldDet.textContent = d.name.toLowerCase();

  // Anima la aguja de la regla tras el primer frame.
  requestAnimationFrame(() => {
    const needle = verdictEl.querySelector(".gauge-needle");
    if (needle) needle.style.left = pct.toFixed(1) + "%";
  });
}

// ── Reiniciar ───────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", reset);

function reset() {
  currentFile = null;
  fileInput.value = "";
  specimenImg.src = "";
  specimen.hidden = true;
  placeholder.hidden = false;
  stopExam();
  clearVerdict();
  analyzeBtn.disabled = true;
  resetBtn.disabled = true;
}
