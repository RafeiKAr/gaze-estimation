/* Eye-Tracking-Demo – Browser-Teil
 *
 * Phasen:  start -> Kalibrierung -> Ergebnis -> Blickverfolgung -> Auswertung
 *
 * Wichtige Einstellungen stehen oben in CONFIG.
 * Alle Positionen werden normiert gespeichert: (0,0) = oben links, (1,1) = unten rechts
 * des Bildschirms, genau wie die Labels in der Masterarbeit.
 */
"use strict";

const CONFIG = {
  calibSeconds: 60,       // Dauer der Kalibrierung (länger = langsamerer Punkt)
  countdown: 3,           // Sekunden Countdown vor der Kalibrierung
  labelsPerSecond: 5,     // Bilder pro Sekunde in der Kalibrierung (Server muss mitkommen)
  margin: 0.05,           // Abstand des Punktes zum Bildschirmrand (Anteil)
  rows: 4,                // Zeilen der Schlangenlinie, danach gleich viele Spalten
                          // (weniger Zeilen = kürzerer Weg = langsamerer Punkt)
  camWidth: 640,          // Kamerabild, das an den Server geht
  camHeight: 480,
  jpegQuality: 0.8,
  smoothing: 0.5,         // Glättung der Blickposition (0 = keine Bewegung, 1 = keine Glättung)
  medianWindow: 3,        // Median über die letzten N Vorhersagen (mehr = ruhiger, aber mehr Verzögerung)
  gridCols: 3,            // Felder des Bühnenlichts: 3 Spalten x 3 Zeilen
  gridRows: 3,            //   (in der Adresse änderbar, z. B. ?grid=7x5 für feinere Felder)
  minGlowPx: 40,          // kleinster Radius des Lichtflecks
};
// Zum Testen: ?calib=10 in der Adresse = Kalibrierung nur 10 Sekunden
// ?rows=3 = langsamer, ?rows=5 = schneller
const urlParams = new URLSearchParams(location.search);
const urlCalib = Number(urlParams.get("calib"));
if (urlCalib > 0) CONFIG.calibSeconds = urlCalib;
const urlRows = Number(urlParams.get("rows"));
if (urlRows >= 2) CONFIG.rows = Math.round(urlRows);
const urlGrid = /^(\d+)x(\d+)$/.exec(urlParams.get("grid") || "");
if (urlGrid) { CONFIG.gridCols = Math.max(1, +urlGrid[1]); CONFIG.gridRows = Math.max(1, +urlGrid[2]); }

// ---------------------------------------------------------------- Hilfen
const $ = (id) => document.getElementById(id);
const pages = ["start", "stage", "result", "analysis"];
function show(page) { pages.forEach((p) => $(p).classList.toggle("hidden", p !== page)); }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

async function enterFullscreen() {
  try { if (!document.fullscreenElement) await document.documentElement.requestFullscreen(); }
  catch (e) { /* Vollbild nicht erlaubt (z.B. im eingebetteten Rahmen) – trotzdem weiter */ }
  await sleep(400);                                   // Größe nach dem Umschalten stabil
}
async function exitFullscreen() {
  try { if (document.fullscreenElement) await document.exitFullscreen(); } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------- Zustand
const state = {
  sid: null,
  calib: null,          // Ergebnis der Kalibrierung
  track: [],            // [{t, px, py, bx, by}]
  tracking: false,
  showBefore: true,
  marker: 0,            // 0 = Bühnenlicht (Felder), 1 = Scheinwerfer (folgt dem Blick), 2 = Lichtpunkt
};
const MARKERS = ["Bühnenlicht (Felder)", "Scheinwerfer (folgt dem Blick)", "Lichtpunkt"];
const NCELLS = CONFIG.gridCols * CONFIG.gridRows;
const cellLight = new Array(NCELLS).fill(0);   // Helligkeit jedes Feldes (0..1), weich überblendet
/** Nummer des feinen Feldes (Spalte, Zeile) für einen Blickpunkt. */
const cellOf = (x, y) => Math.min(CONFIG.gridRows - 1, Math.max(0, Math.floor(y * CONFIG.gridRows))) * CONFIG.gridCols +
  Math.min(CONFIG.gridCols - 1, Math.max(0, Math.floor(x * CONFIG.gridCols)));
const cellName = (k) => `S${(k % CONFIG.gridCols) + 1}/Z${Math.floor(k / CONFIG.gridCols) + 1}`;

// ---------------------------------------------------------------- Kamera
const cam = $("cam");
const grab = document.createElement("canvas");
grab.width = CONFIG.camWidth; grab.height = CONFIG.camHeight;
const grabCtx = grab.getContext("2d");

let camLabel = "?";
let imageCapture = null;            // liest Bilder direkt aus der Kamera (Chrome/Edge)
async function startCamera(deviceId) {
  if (cam.srcObject) cam.srcObject.getTracks().forEach((t) => t.stop());
  const video = { width: { ideal: CONFIG.camWidth }, height: { ideal: CONFIG.camHeight } };
  if (deviceId) video.deviceId = { exact: deviceId }; else video.facingMode = "user";
  const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  const track = stream.getVideoTracks()[0];
  camLabel = track.label || "Kamera";
  imageCapture = "ImageCapture" in window ? new ImageCapture(track) : null;
  cam.srcObject = stream;
  $("preview").srcObject = stream;
  await new Promise((r) => (cam.readyState >= 2 ? r() : (cam.onloadeddata = r)));
  await cam.play();                 // ausdrücklich starten (nicht per autoplay)
  // Seitenverhältnis der Kamera beibehalten (Breite 640)
  const s = track.getSettings();
  const vw = s.width || cam.videoWidth, vh = s.height || cam.videoHeight;
  if (vw && vh) {
    grab.width = CONFIG.camWidth;
    grab.height = Math.round((CONFIG.camWidth * vh) / vw);
  }
}

/* Prüfen, ob die Kamera brauchbare Bilder liefert:
 *  - schwarz: mittlere Helligkeit fast 0 (Browser zeichnet das Video nicht)
 *  - eingefroren: dasselbe JPEG mehrmals hintereinander */
const camCheck = { last: "", same: 0, dark: 0 };
function cameraFrozen() { return camCheck.same >= 6 || camCheck.dark >= 3; }
const probe = document.createElement("canvas");
probe.width = 32; probe.height = 24;
const probeCtx = probe.getContext("2d", { willReadFrequently: true });

/** Aktuelles Kamerabild als JPEG (NICHT gespiegelt, wie die Aufnahmen der Masterarbeit).
 *  Gibt null zurück, wenn das Bild schwarz ist (wird dann nicht gesendet). */
let grabQueue = Promise.resolve();
function grabFrame() {                        // immer nur ein Kamerazugriff gleichzeitig
  const p = grabQueue.then(grabFrameNow);
  grabQueue = p.catch(() => null);
  return p;
}
let lastBrightness = -1;
function drawAndMeasure(src) {
  grabCtx.drawImage(src, 0, 0, grab.width, grab.height);
  probeCtx.drawImage(grab, 0, 0, probe.width, probe.height);
  const px = probeCtx.getImageData(0, 0, probe.width, probe.height).data;
  let sum = 0;
  for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
  return sum / (3 * probe.width * probe.height);
}
async function grabFrameNow() {
  let brightness = -1;
  if (imageCapture) {                                  // 1. direkt aus der Kamera
    try {
      const bmp = await imageCapture.grabFrame();
      brightness = drawAndMeasure(bmp);
      bmp.close();
    } catch (e) { brightness = -1; }
  }
  if (brightness < 4) {                                // 2. Ersatz: aus dem Video-Element
    if (cam.paused) cam.play().catch(() => {});
    if (cam.readyState >= 2) brightness = drawAndMeasure(cam);
  }
  if (brightness < 4 && $("preview").readyState >= 2) {  // 3. Ersatz: Vorschau-Video
    brightness = drawAndMeasure($("preview"));
  }
  lastBrightness = brightness;
  camCheck.dark = brightness < 4 ? camCheck.dark + 1 : 0;
  if (brightness < 4) return null;
  const url = grab.toDataURL("image/jpeg", CONFIG.jpegQuality);
  camCheck.same = url === camCheck.last ? camCheck.same + 1 : 0;
  camCheck.last = url;
  return url;
}

// ---------------------------------------------------------------- Bühne (Canvas)
const stage = $("stage-canvas");
const sctx = stage.getContext("2d");
function resizeStage() {
  const dpr = window.devicePixelRatio || 1;
  stage.width = Math.round(window.innerWidth * dpr);
  stage.height = Math.round(window.innerHeight * dpr);
  sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resizeStage);
const W = () => window.innerWidth;
const H = () => window.innerHeight;

function drawGrid() {
  sctx.fillStyle = "#0b1020";
  sctx.fillRect(0, 0, W(), H());
  sctx.strokeStyle = "rgba(255,255,255,0.05)";
  sctx.lineWidth = 1;
  for (let i = 1; i < 3; i++) {
    sctx.beginPath(); sctx.moveTo((W() * i) / 3, 0); sctx.lineTo((W() * i) / 3, H()); sctx.stroke();
    sctx.beginPath(); sctx.moveTo(0, (H() * i) / 3); sctx.lineTo(W(), (H() * i) / 3); sctx.stroke();
  }
}

// ---------------------------------------------------------------- Kalibrierpfad
/* Schlangenlinie: erst waagerechte Zeilen von oben nach unten, dann senkrechte
 * Spalten von rechts nach links. Der Punkt bewegt sich mit konstanter Geschwindigkeit
 * und deckt so den ganzen Bildschirm inkl. Ränder und Ecken ab (wie Gen 1). */
function buildPath() {
  const m = CONFIG.margin, n = CONFIG.rows, pts = [];
  const lin = (i) => m + ((1 - 2 * m) * i) / (n - 1);
  for (let r = 0; r < n; r++) {
    const y = lin(r);
    if (r % 2 === 0) pts.push([m, y], [1 - m, y]); else pts.push([1 - m, y], [m, y]);
  }
  for (let c = n - 1; c >= 0; c--) {
    const x = lin(c);
    if ((n - 1 - c) % 2 === 0) pts.push([x, 1 - m], [x, m]); else pts.push([x, m], [x, 1 - m]);
  }
  // Länge in Pixeln (damit die Geschwindigkeit auf dem Bildschirm konstant ist)
  const seg = [];
  let total = 0;
  for (let i = 1; i < pts.length; i++) {
    const dx = (pts[i][0] - pts[i - 1][0]) * W(), dy = (pts[i][1] - pts[i - 1][1]) * H();
    const len = Math.hypot(dx, dy);
    seg.push(len); total += len;
  }
  return { pts, seg, total };
}

function pointOnPath(path, frac) {
  let d = Math.min(Math.max(frac, 0), 1) * path.total;
  for (let i = 0; i < path.seg.length; i++) {
    if (d <= path.seg[i] || i === path.seg.length - 1) {
      const a = path.pts[i], b = path.pts[i + 1], f = path.seg[i] ? d / path.seg[i] : 0;
      return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
    }
    d -= path.seg[i];
  }
  return path.pts[path.pts.length - 1];
}

function drawDot(x, y, pulse) {
  const px = x * W(), py = y * H(), r = 12 + 3 * Math.sin(pulse);
  // Gleiches Licht wie bei der Verfolgung (ein Feld des Bühnenlichts um den Punkt).
  // So spiegelt sich in Kalibrierung und Verfolgung dasselbe helle Feld in Augen
  // und Brille -> die Bilder beider Phasen passen zusammen.
  const cw = W() / CONFIG.gridCols, ch = H() / CONFIG.gridRows;
  softLight(px - cw / 2, py - ch / 2, cw, ch, 0.8);
  sctx.beginPath(); sctx.arc(px, py, r + 10, 0, 2 * Math.PI);
  sctx.fillStyle = "rgba(255,60,60,0.18)"; sctx.fill();
  sctx.beginPath(); sctx.arc(px, py, r, 0, 2 * Math.PI);
  sctx.fillStyle = "#ff3c3c"; sctx.fill();
  sctx.beginPath(); sctx.arc(px, py, 3, 0, 2 * Math.PI);
  sctx.fillStyle = "#fff"; sctx.fill();
}

// ---------------------------------------------------------------- Phase 1: Kalibrierung
async function runCalibration() {
  if (!state.sid) state.sid = (await api("/api/session")).sid;
  else { await api("/api/end", { sid: state.sid }); state.sid = (await api("/api/session")).sid; }
  state.calib = null;

  show("stage");
  await enterFullscreen();
  resizeStage();
  const path = buildPath();
  $("stage-hud").innerHTML = "<b>Kalibrierung</b> – folge dem roten Punkt nur mit den Augen.";
  $("stage-bar").style.width = "0%";

  // Countdown: Punkt steht am Start
  for (let c = CONFIG.countdown; c > 0; c--) {
    drawGrid(); const p = pointOnPath(path, 0); drawDot(p[0], p[1], 0);
    $("stage-center").textContent = `Schau auf den roten Punkt … ${c}`;
    await sleep(1000);
  }
  $("stage-center").textContent = "";

  const pending = [];
  let sent = 0, failed = 0, frozen = 0;
  const t0 = performance.now();
  const dur = CONFIG.calibSeconds * 1000;

  // Bilder mit fester Rate aufnehmen; Label = Punktposition im Moment der Aufnahme
  const timer = setInterval(() => {
    const t = (performance.now() - t0) / 1000;
    if (t * 1000 > dur) return;
    const [x, y] = pointOnPath(path, (t * 1000) / dur);  // Label = Position beim Auslösen
    const job = grabFrame().then((image) => {
      if (cameraFrozen()) {
        $("stage-hud").innerHTML = "<b>⚠ Die Kamera liefert keine brauchbaren Bilder (schwarz oder eingefroren).</b>";
      }
      if (!image) { frozen++; return; }
      return api("/api/calib_frame", { sid: state.sid, image, x, y, t })
        .then(() => { sent++; }).catch(() => { failed++; });
    });
    pending.push(job);
  }, 1000 / CONFIG.labelsPerSecond);

  // Animation des Punktes
  await new Promise((resolve) => {
    function frame(now) {
      const el = now - t0;
      drawGrid();
      const [x, y] = pointOnPath(path, el / dur);
      drawDot(x, y, el / 120);
      $("stage-bar").style.width = `${Math.min(100, (100 * el) / dur)}%`;
      if (el < dur) requestAnimationFrame(frame); else resolve();
    }
    requestAnimationFrame(frame);
  });
  clearInterval(timer);

  drawGrid();
  $("stage-hud").innerHTML = "";
  $("stage-center").textContent = "Bilder werden übertragen …";
  await Promise.all(pending);
  if (sent < 20 || frozen > sent) {
    await exitFullscreen();
    alert(`Die Kamera hat keine brauchbaren Bilder geliefert (gesendet: ${sent}, schwarz/eingefroren: ${frozen}).\n` +
      "Bitte auf der Startseite eine andere Kamera wählen, die Seite neu laden (Strg+F5) und Chrome oder Edge verwenden.");
    show("start");
    return;
  }
  $("stage-center").textContent = "Das Modell wird an dich angepasst …";
  try {
    state.calib = await api("/api/calibrate", { sid: state.sid });
  } catch (e) {
    await exitFullscreen();
    alert(`Kalibrierung fehlgeschlagen: ${e.message}\n(übertragen: ${sent}, Fehler: ${failed}, eingefrorene Kamerabilder: ${frozen})`);
    show("start");
    return;
  }
  $("stage-center").textContent = "";
  await runAccuracyTest();
  await exitFullscreen();
  showResult();
}

// ---------------------------------------------------------------- Genauigkeitstest + Offset-Korrektur
/* Nach der Kalibrierung: 5 feste Punkte je 2,4 s. Die Vorhersagen der letzten 1,6 s
 * je Punkt werden mit der Punktposition verglichen. Daraus:
 *  - ehrlicher Testfehler (diese Bilder kamen im Training nicht vor),
 *  - systematischer Versatz (Offset) -> wird bei der Verfolgung abgezogen
 *    (wie die Offset-Kalibrierung von Chen & Shi im Stand der Technik).
 * Ob die Korrektur hilft, wird fair geprüft: Für jeden Punkt wird der Versatz nur
 * aus den ANDEREN vier Punkten geschätzt (Kreuzvalidierung). */
const TEST_MS = 3000;                     // so lange wird jeder Testpunkt gezeigt
const TEST_POINTS = [[0.5, 0.5], [0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]];
const ediag = (pairs) => 100 * Math.sqrt(pairs.reduce((s, [p, t]) =>
  s + (p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2, 0) / Math.max(1, pairs.length)) / Math.SQRT2;

async function runAccuracyTest() {
  $("stage-hud").innerHTML = "<b>Kurzer Test</b> – schau auf jeden Punkt, bis er verschwindet.";
  const perPoint = [];
  for (const [tx, ty] of TEST_POINTS) {
    const preds = [], t0 = performance.now();
    let running = true;
    const anim = () => {
      if (!running) return;
      const el = (performance.now() - t0) / TEST_MS;
      drawGrid(); drawDot(tx, ty, el * 8);
      sctx.strokeStyle = "rgba(255,255,255,0.5)"; sctx.lineWidth = 2;
      sctx.beginPath(); sctx.arc(tx * W(), ty * H(), 34, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * (1 - el)); sctx.stroke();
      requestAnimationFrame(anim);
    };
    requestAnimationFrame(anim);
    while (performance.now() - t0 < TEST_MS) {
      const image = await grabFrame();
      if (!image) { await sleep(100); continue; }
      try {
        // Erst ehrlich vorhersagen, dann merkt sich der Server das Bild mit Ziel
        // (keep) für das Nachtraining. Die ersten 0,8 s zählen nicht (Blick springt noch).
        const keep = performance.now() - t0 > 800;
        const r = await api("/api/test_frame", { sid: state.sid, image, x: tx, y: ty, keep });
        if (keep && r.personal) preds.push(r.personal);
      } catch (e) { await sleep(200); }
    }
    running = false;
    perPoint.push({ t: [tx, ty], preds });
  }
  const used = perPoint.filter((p) => p.preds.length);
  const mean = (arr, k) => arr.reduce((s, v) => s + v[k], 0) / arr.length;
  const off = (pts) => [mean(pts.map((p) => [mean(p.preds, 0) - p.t[0], mean(p.preds, 1) - p.t[1]]), 0),
                        mean(pts.map((p) => [mean(p.preds, 0) - p.t[0], mean(p.preds, 1) - p.t[1]]), 1)];
  const raw = [], cv = [];
  used.forEach((p, i) => {
    const o = used.length > 1 ? off(used.filter((_, j) => j !== i)) : [0, 0];
    p.preds.forEach((q) => { raw.push([q, p.t]); cv.push([[q[0] - o[0], q[1] - o[1]], p.t]); });
  });
  state.test = { raw: raw.length ? ediag(raw) : null, corrected: cv.length ? ediag(cv) : null, n: raw.length };
  state.bias = used.length >= 3 && state.test.corrected < state.test.raw ? off(used) : [0, 0];
  state.bias = state.bias.map((b) => Math.max(-0.15, Math.min(0.15, b)));   // höchstens 15 %
  state.testPoints = used.map((p) => ({ target: p.t, preds: p.preds }));
  state.test.applied = state.bias[0] !== 0 || state.bias[1] !== 0;
  // Nachtraining: Ridge mit Kalibrier- UND Testbildern neu rechnen. Danach ist die
  // Offset-Korrektur nicht mehr nötig (der Versatz steckt schon im neuen Modell).
  try {
    const rf = await api("/api/refit", { sid: state.sid });
    state.test.refit = !!rf.refit;
    if (rf.refit) { state.bias = [0, 0]; state.test.applied = false; }
  } catch (e) { state.test.refit = false; }
  $("stage-hud").innerHTML = "";
}

// ---------------------------------------------------------------- Ergebnis
function showResult() {
  const c = state.calib;
  const rows = [
    ["Konstante Schätzung (Mittelwert)", c.error_constant, "#4a5578"],
    ["Vorher: Basismodell", c.error_before, "#8a93a8"],
    ["Nachher: personalisiert", c.error_after, "#ffb347"],
  ];
  const max = Math.max(...rows.map((r) => r[1]), 1);
  $("bars").innerHTML = rows.map(([name, v, col]) => `
    <div class="bar-row"><div>${name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${(100 * v) / max}%;background:${col}"></div></div>
      <div class="bar-val">${v.toFixed(1)} %</div></div>`).join("");
  const red = c.error_before > 0 ? (100 * (1 - c.error_after / c.error_before)).toFixed(0) : "–";
  $("result-details").textContent =
    `Verbesserung gegenüber dem Basismodell: ${red} %. ` +
    `${c.frames} Bilder verwendet (${c.train_frames} Training, ${c.val_frames} Prüfung), ` +
    `beste Epoche ${c.best_epoch}, Training ${c.seconds} s.` +
    (c.best_epoch === 0 ? " Hinweis: Die Anpassung hat nichts verbessert – bitte bei besserem Licht neu kalibrieren." : "");
  const t = state.test || {};
  $("result-test").innerHTML = t.raw == null ? "" :
    `<b>Test mit 5 Punkten</b> (nicht im Training): Fehler <b>${t.raw.toFixed(1)} %</b>` +
    (t.refit ? ". Die Testbilder wurden danach <b>zusätzlich ins Training übernommen</b> (Nachtraining)"
     : t.applied ? ` → <b>${t.corrected.toFixed(1)} %</b> mit Offset-Korrektur (wird bei der Verfolgung verwendet)`
               : " (eine Offset-Korrektur würde nicht helfen)") + `. ${t.n} Messungen.`;
  $("result-diag").textContent =
    `Diagnose: Trainingsfehler ${c.train_before} % → ${c.train_last_epoch} % (letzte Epoche), ` +
    `Bildänderung ${c.feature_change}, Streuung der Ziele ${c.target_spread}, ` +
    `Kamera ${grab.width}×${grab.height} (${camLabel}), ` +
    `Ausschnitt: ${c.crop}` + (c.face_box ? ` (${c.face_box.join(",")})` : "") + ". " +
    (c.method ? `Verfahren: ${{ mlp: "MLP-Kopf", ridge: "Ridge (global)", spatial: "Ridge (räumlich)" }[c.method]} ` +
      `(MLP ${c.error_mlp} %, Ridge global ${c.error_ridge} %, Ridge räumlich ${c.error_spatial} %). ` +
      `Gewicht Kopfposition: ${c.head_weight_spatial ?? "–"} (räumlich), ${c.head_weight_ridge ?? "–"} (global).` : "");
  show("result");
}

// ---------------------------------------------------------------- Phase 2: Blickverfolgung
/** Fehler für die Größe des Lichts: bevorzugt der ehrliche 5-Punkte-Test. */
function currentErr() {
  const t = state.test || {};
  const v = t.applied ? t.corrected : t.raw;
  return (v != null ? v : (state.calib.error_after || 20)) / 100;
}
const smooth = { px: null, py: null, bx: null, by: null };
const recent = [];                                // letzte Rohwerte für den Median-Filter
function filtered(p) {
  recent.push(p); if (recent.length > CONFIG.medianWindow) recent.shift();
  const med = (k) => { const s = recent.map((q) => q[k]).sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  const b = state.bias || [0, 0];
  return [Math.min(1, Math.max(0, med(0) - b[0])), Math.min(1, Math.max(0, med(1) - b[1]))];
}
const latest = { personal: null, base: null };

function glow(x, y, radius) {
  const g = sctx.createRadialGradient(x, y, 0, x, y, radius);
  g.addColorStop(0, "rgba(255,236,170,0.95)");
  g.addColorStop(0.25, "rgba(255,179,71,0.65)");
  g.addColorStop(0.6, "rgba(255,122,89,0.22)");
  g.addColorStop(1, "rgba(255,122,89,0)");
  sctx.fillStyle = g;
  sctx.beginPath(); sctx.arc(x, y, radius, 0, 2 * Math.PI); sctx.fill();
  sctx.beginPath(); sctx.arc(x, y, 5, 0, 2 * Math.PI); sctx.fillStyle = "#fff"; sctx.fill();
}

/** Weiches, warmes Rechteck-Licht (wie ein Scheinwerfer auf der Bühne). */
function softLight(x0, y0, w, h, intensity) {
  if (intensity < 0.01) return;
  sctx.save();
  sctx.filter = `blur(${Math.round(Math.min(w, h) * 0.12)}px)`;
  const g = sctx.createRadialGradient(x0 + w / 2, y0 + h / 2, 0, x0 + w / 2, y0 + h / 2, Math.hypot(w, h) / 2);
  g.addColorStop(0, `rgba(255,244,214,${0.88 * intensity})`);
  g.addColorStop(0.7, `rgba(255,220,160,${0.62 * intensity})`);
  g.addColorStop(1, `rgba(255,196,128,${0.34 * intensity})`);
  sctx.fillStyle = g;
  sctx.fillRect(x0 + w * 0.04, y0 + h * 0.04, w * 0.92, h * 0.92);
  sctx.restore();
}

/** Bühnenlicht: Das Feld (Standard 7 x 5), in dem der Blick liegt, wird schnell hell,
 *  die Nachbarfelder leuchten schwach mit, die anderen werden dunkel. */
function stageLights(x, y) {
  const C = CONFIG.gridCols, R = CONFIG.gridRows, w = W() / C, h = H() / R;
  const cur = cellOf(x, y), cc = cur % C, cr = Math.floor(cur / C);
  for (let k = 0; k < NCELLS; k++) {
    const d = Math.max(Math.abs((k % C) - cc), Math.abs(Math.floor(k / C) - cr));
    const goal = d === 0 ? 1 : d === 1 ? 0.18 : 0;           // Nachbarn leicht mit beleuchten
    cellLight[k] += (goal - cellLight[k]) * 0.3;
    softLight((k % C) * w, Math.floor(k / C) * h, w, h, cellLight[k]);
  }
  sctx.fillStyle = `rgba(255,235,200,${0.35 + 0.5 * cellLight[cur]})`;
  sctx.font = "600 16px Segoe UI, sans-serif";
  sctx.fillText(REGIONS[regionOf(x, y)], cc * w + 12, cr * h + 26);
}

/** Scheinwerfer: Rechteck-Licht folgt dem Blick; Größe = gemessene Unsicherheit. */
function spotLight(x, y) {
  const err = currentErr();
  const w = Math.max(W() * 0.18, 2.2 * err * W()), h = Math.max(H() * 0.18, 2.2 * err * H());
  softLight(x * W() - w / 2, y * H() - h / 2, w, h, 1);
}

function drawTracking() {
  if (!state.tracking) return;
  sctx.fillStyle = "#04060b";                             // dunkle Bühne
  sctx.fillRect(0, 0, W(), H());
  sctx.strokeStyle = "rgba(255,255,255,0.05)"; sctx.lineWidth = 1;
  for (let i = 1; i < CONFIG.gridCols; i++) {
    const x = (W() * i) / CONFIG.gridCols;
    sctx.beginPath(); sctx.moveTo(x, 0); sctx.lineTo(x, H()); sctx.stroke();
  }
  for (let i = 1; i < CONFIG.gridRows; i++) {
    const y = (H() * i) / CONFIG.gridRows;
    sctx.beginPath(); sctx.moveTo(0, y); sctx.lineTo(W(), y); sctx.stroke();
  }
  const a = CONFIG.smoothing;
  const lerp = (o, n) => (o === null ? n : o + a * (n - o));
  if (latest.personal) {
    smooth.px = lerp(smooth.px, latest.personal[0]); smooth.py = lerp(smooth.py, latest.personal[1]);
  }
  if (latest.base) {
    smooth.bx = lerp(smooth.bx, latest.base[0]); smooth.by = lerp(smooth.by, latest.base[1]);
  }
  const diag = Math.hypot(W(), H());
  if (state.showBefore && smooth.bx !== null) {           // Vorher: grauer Ring
    const x = smooth.bx * W(), y = smooth.by * H();
    sctx.strokeStyle = "rgba(170,180,200,0.9)"; sctx.lineWidth = 3;
    sctx.beginPath(); sctx.arc(x, y, 16, 0, 2 * Math.PI); sctx.stroke();
    sctx.fillStyle = "rgba(170,180,200,0.9)"; sctx.font = "13px Segoe UI, sans-serif";
    sctx.fillText("vorher", x + 22, y + 4);
  }
  if (smooth.px !== null) {                                // Nachher: Licht
    if (state.marker === 0) stageLights(smooth.px, smooth.py);
    else if (state.marker === 1) spotLight(smooth.px, smooth.py);
    else glow(smooth.px * W(), smooth.py * H(),
      Math.max(CONFIG.minGlowPx, currentErr() * diag));
  }
  if (cameraFrozen()) {
    sctx.fillStyle = "#ff6b6b"; sctx.font = "600 20px Segoe UI, sans-serif";
    sctx.fillText("⚠ Kamera liefert keine neuen Bilder – Esc und Seite neu laden", 20, H() - 30);
  }
  requestAnimationFrame(drawTracking);
}

async function runTracking() {
  show("stage");
  await enterFullscreen();
  resizeStage();
  $("stage-bar").style.width = "0%";
  $("stage-center").textContent = "";
  $("stage-hud").innerHTML =
    `<b>Blickverfolgung</b> · Fehler vorher <b>${state.calib.error_before.toFixed(1)} %</b>, ` +
    `nachher <b>${state.calib.error_after.toFixed(1)} %</b><br>` +
    `Licht = nachher · grauer Ring = vorher<br>` +
    `<b>M</b> Lichtart wechseln (Bühnenlicht / Scheinwerfer / Lichtpunkt) · <b>V</b> Vergleich ein/aus · <b>Esc</b> Beenden`;
  cellLight.fill(0);
  state.tracking = true;
  smooth.px = smooth.py = smooth.bx = smooth.by = null;
  recent.length = 0;
  requestAnimationFrame(drawTracking);
  const t0 = performance.now() - (state.track.length ? state.track[state.track.length - 1].t * 1000 : 0);

  // Immer nur eine Anfrage gleichzeitig. Das nächste Kamerabild wird schon geholt,
  // während der Server noch rechnet (weniger Verzögerung).
  let next = grabFrame();
  while (state.tracking) {
    try {
      const image = await next;
      next = grabFrame();
      if (!image) { await sleep(200); continue; }
      const r = await api("/api/predict", { sid: state.sid, image });
      latest.base = r.base;
      latest.personal = r.personal ? filtered(r.personal) : null;
      if (latest.personal) {
        state.track.push({ t: (performance.now() - t0) / 1000,
          px: latest.personal[0], py: latest.personal[1], bx: r.base[0], by: r.base[1] });
      }
    } catch (e) {
      $("stage-center").textContent = `Verbindung unterbrochen: ${e.message}`;
      await sleep(1000);
    }
  }
}

async function stopTracking() {
  if (!state.tracking) return;
  state.tracking = false;
  // Nur bei GAZE_DEBUG=1 speichert der Server diesen Bericht (für die Fehlersuche)
  api("/api/debug_log", { sid: state.sid, data: { test: state.test, bias: state.bias,
    test_points: state.testPoints, track: state.track.slice(0, 600) } }).catch(() => {});
  await exitFullscreen();
  showAnalysis();
}

document.addEventListener("keydown", (e) => {
  if (!state.tracking) return;
  if (e.key === "v" || e.key === "V") state.showBefore = !state.showBefore;
  if (e.key === "m" || e.key === "M") state.marker = (state.marker + 1) % MARKERS.length;
  if (e.key === "Escape") stopTracking();
});
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && state.tracking) stopTracking();
});

// ---------------------------------------------------------------- Phase 3: Auswertung
const REGIONS = ["oben links", "oben Mitte", "oben rechts", "Mitte links", "Mitte",
  "Mitte rechts", "unten links", "unten Mitte", "unten rechts"];
const regionOf = (x, y) => Math.min(2, Math.max(0, Math.floor(y * 3))) * 3 +
  Math.min(2, Math.max(0, Math.floor(x * 3)));

// Farbskala „magma“ (dunkel -> hell)
const MAGMA = [[0, [0, 0, 4]], [0.25, [59, 15, 112]], [0.5, [140, 41, 129]],
  [0.75, [222, 73, 104]], [0.9, [254, 159, 109]], [1, [252, 253, 191]]];
function magma(v) {
  for (let i = 1; i < MAGMA.length; i++) {
    if (v <= MAGMA[i][0]) {
      const [a, ca] = MAGMA[i - 1], [b, cb] = MAGMA[i], f = (v - a) / (b - a);
      return ca.map((c, k) => Math.round(c + f * (cb[k] - c)));
    }
  }
  return MAGMA[MAGMA.length - 1][1];
}

function drawHeatmap(canvas, points, title) {
  const aspect = (window.screen.height || 9) / (window.screen.width || 16);
  canvas.width = 640; canvas.height = Math.round(640 * aspect);
  const ctx = canvas.getContext("2d");
  const gw = 160, gh = Math.round(160 * aspect), grid = new Float32Array(gw * gh);
  const sig = 5, rad = 3 * sig;
  for (const [x, y] of points) {
    const cx = x * gw, cy = y * gh;
    for (let j = Math.max(0, Math.floor(cy - rad)); j < Math.min(gh, cy + rad); j++)
      for (let i = Math.max(0, Math.floor(cx - rad)); i < Math.min(gw, cx + rad); i++)
        grid[j * gw + i] += Math.exp(-((i - cx) ** 2 + (j - cy) ** 2) / (2 * sig * sig));
  }
  const mx = Math.max(...grid, 1e-9);
  const img = new ImageData(gw, gh);
  for (let k = 0; k < grid.length; k++) {
    const v = Math.sqrt(grid[k] / mx), c = magma(v);
    img.data.set([c[0], c[1], c[2], 255], k * 4);
  }
  const tmp = document.createElement("canvas"); tmp.width = gw; tmp.height = gh;
  tmp.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";             // feine Felder (7 x 5)
  for (let i = 1; i < CONFIG.gridCols; i++) {
    ctx.beginPath(); ctx.moveTo((canvas.width * i) / CONFIG.gridCols, 0); ctx.lineTo((canvas.width * i) / CONFIG.gridCols, canvas.height); ctx.stroke();
  }
  for (let i = 1; i < CONFIG.gridRows; i++) {
    ctx.beginPath(); ctx.moveTo(0, (canvas.height * i) / CONFIG.gridRows); ctx.lineTo(canvas.width, (canvas.height * i) / CONFIG.gridRows); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.25)";             // grobe Bereiche (3 x 3)
  for (let i = 1; i < 3; i++) {
    ctx.beginPath(); ctx.moveTo((canvas.width * i) / 3, 0); ctx.lineTo((canvas.width * i) / 3, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, (canvas.height * i) / 3); ctx.lineTo(canvas.width, (canvas.height * i) / 3); ctx.stroke();
  }
  if (title) { ctx.fillStyle = "#fff"; ctx.font = "16px Segoe UI, sans-serif"; ctx.fillText(title, 10, 22); }
}

function drawTimeline(canvas, track) {
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth || 1000, ch = 260;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0e1426"; ctx.fillRect(0, 0, cw, ch);
  const left = 110, right = 16, top = 10, bottom = 30, rowH = (ch - top - bottom) / 9;
  const T = Math.max(1, track.length ? track[track.length - 1].t : 1);
  const X = (t) => left + ((cw - left - right) * t) / T;
  ctx.font = "12px Segoe UI, sans-serif";
  REGIONS.forEach((name, r) => {
    const y = top + r * rowH;
    ctx.fillStyle = r % 2 ? "#121a30" : "#0e1426"; ctx.fillRect(left, y, cw - left - right, rowH);
    ctx.fillStyle = "#9aa4bf"; ctx.fillText(name, 8, y + rowH * 0.65);
  });
  ctx.fillStyle = "#ffb347";                              // zusammenhängende Abschnitte = ein Balken
  let i = 0;
  while (i < track.length) {
    const r = regionOf(track[i].px, track[i].py), start = track[i].t;
    let j = i;
    while (j + 1 < track.length && regionOf(track[j + 1].px, track[j + 1].py) === r) j++;
    const end = j + 1 < track.length ? track[j + 1].t : T;
    ctx.fillRect(X(start), top + r * rowH + 2, Math.max(1.5, X(Math.min(end, T)) - X(start)), rowH - 4);
    i = j + 1;
  }
  ctx.fillStyle = "#9aa4bf";                              // Zeitachse
  const step = T > 120 ? 30 : T > 40 ? 10 : 5;
  for (let s = 0; s <= T; s += step) {
    ctx.fillText(`${s} s`, X(s) - 8, ch - 10);
    ctx.fillRect(X(s), ch - bottom, 1, 4);
  }
}

/** Verlauf der Blickposition: x (links -> rechts) und y (oben -> unten) über die Zeit. */
function drawPositionChart(canvas, track) {
  const dpr = window.devicePixelRatio || 1, cw = canvas.clientWidth || 1000, ch = 240;
  canvas.width = cw * dpr; canvas.height = ch * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#0e1426"; ctx.fillRect(0, 0, cw, ch);
  const left = 110, right = 16, top = 14, bottom = 30, ph = ch - top - bottom;
  const T = Math.max(1, track.length ? track[track.length - 1].t : 1);
  const X = (t) => left + ((cw - left - right) * t) / T, Y = (v) => top + ph * (1 - v);
  ctx.font = "12px Segoe UI, sans-serif"; ctx.fillStyle = "#9aa4bf";
  [[0, "0 %"], [0.5, "50 %"], [1, "100 %"]].forEach(([v, s]) => {
    ctx.fillText(s, left - 40, Y(v) + 4);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.beginPath(); ctx.moveTo(left, Y(v)); ctx.lineTo(cw - right, Y(v)); ctx.stroke();
  });
  const series = [["px", "#ffb347", "x: links → rechts"], ["py", "#6cb6ff", "y: oben → unten"]];
  series.forEach(([key, col, name], k) => {
    ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
    track.forEach((p, i) => { const fn = i ? "lineTo" : "moveTo"; ctx[fn](X(p.t), Y(p[key])); });
    ctx.stroke();
    ctx.fillStyle = col; ctx.fillRect(cw - right - 150, top + 8 + k * 18, 12, 3);
    ctx.fillStyle = "#c9d1e6"; ctx.fillText(name, cw - right - 134, top + 13 + k * 18);
  });
  ctx.fillStyle = "#9aa4bf";
  const step = T > 120 ? 30 : T > 40 ? 10 : 5;
  for (let s = 0; s <= T; s += step) ctx.fillText(`${s} s`, X(s) - 8, ch - 10);
}

/** Zeittabelle: für jeden Zeitabschnitt die ungefähre Blickposition. */
function timeTable(track) {
  const T = track.length ? track[track.length - 1].t : 0;
  const win = T <= 60 ? 2 : Math.ceil(T / 30);           // höchstens ca. 30 Zeilen
  const rows = [];
  for (let a = 0; a < T + 1e-9; a += win) {
    const pts = track.filter((p) => p.t >= a && p.t < a + win);
    if (!pts.length) continue;
    const med = (arr) => { const s = [...arr].sort((u, v) => u - v); return s[Math.floor(s.length / 2)]; };
    const x = med(pts.map((p) => p.px)), y = med(pts.map((p) => p.py));
    const count = new Array(9).fill(0); pts.forEach((p) => count[regionOf(p.px, p.py)]++);
    const reg = count.indexOf(Math.max(...count));
    rows.push({ from: a, to: Math.min(a + win, T), x, y, region: REGIONS[reg], cell: cellName(cellOf(x, y)),
      share: (100 * count[reg]) / pts.length, n: pts.length });
  }
  return rows;
}

function regionShares(track, key) {
  const dur = new Array(9).fill(0);
  for (let i = 0; i < track.length; i++) {
    const p = track[i], dt = i + 1 < track.length ? track[i + 1].t - p.t : 0.2;
    dur[regionOf(p[key + "x"], p[key + "y"])] += Math.min(dt, 1);   // Pausen begrenzen
  }
  const total = dur.reduce((a, b) => a + b, 0) || 1;
  return dur.map((d) => (100 * d) / total);
}

function showAnalysis() {
  const tr = state.track;
  show("analysis");
  const dur = tr.length ? tr[tr.length - 1].t : 0;
  $("analysis-summary").textContent =
    `${tr.length} Messpunkte in ${dur.toFixed(0)} s (ca. ${(tr.length / Math.max(dur, 1)).toFixed(1)} pro Sekunde). ` +
    `Fehler laut Kalibrierung: vorher ${state.calib.error_before.toFixed(1)} %, ` +
    `nachher ${state.calib.error_after.toFixed(1)} % der Bildschirmdiagonale.`;
  drawHeatmap($("heat-before"), tr.map((p) => [p.bx, p.by]));
  drawHeatmap($("heat-after"), tr.map((p) => [p.px, p.py]));
  drawTimeline($("timeline"), tr);
  drawPositionChart($("position-chart"), tr);
  state.timeRows = timeTable(tr);
  $("time-table").innerHTML = "<table><tr><th>Zeitraum</th><th>Bereich</th><th>Feld</th>" +
    "<th class='num'>x (% Breite)</th><th class='num'>y (% Höhe)</th><th class='num'>sicher</th></tr>" +
    state.timeRows.map((r) => `<tr><td>${r.from.toFixed(0)}–${r.to.toFixed(0)} s</td><td>${r.region}</td><td>${r.cell}</td>` +
      `<td class="num">${(100 * r.x).toFixed(0)}</td><td class="num">${(100 * r.y).toFixed(0)}</td>` +
      `<td class="num">${r.share.toFixed(0)} %</td></tr>`).join("") + "</table>";
  const after = regionShares(tr, "p"), before = regionShares(tr, "b");
  $("region-table").innerHTML = "<table><tr><th>Bereich</th><th class='num'>Nachher</th>" +
    "<th class='num'>Vorher</th></tr>" + REGIONS.map((n, i) =>
      `<tr><td>${n}</td><td class="num">${after[i].toFixed(1)} %</td><td class="num">${before[i].toFixed(1)} %</td></tr>`
    ).join("") + "</table>";
}

function download(name, url) {
  const a = document.createElement("a"); a.href = url; a.download = name; a.click();
}

function exportPng() {
  const hb = $("heat-before"), ha = $("heat-after"), tl = $("timeline");
  const pad = 20, w = hb.width * 2 + pad * 3, tlh = Math.round((tl.height / tl.width) * (w - 2 * pad));
  const out = document.createElement("canvas");
  out.width = w; out.height = 60 + hb.height + 40 + tlh + pad;
  const c = out.getContext("2d");
  c.fillStyle = "#0b1020"; c.fillRect(0, 0, out.width, out.height);
  c.fillStyle = "#e8ecf6"; c.font = "22px Segoe UI, sans-serif";
  c.fillText(`Blickauswertung – Fehler vorher ${state.calib.error_before.toFixed(1)} %, ` +
    `nachher ${state.calib.error_after.toFixed(1)} %`, pad, 36);
  c.drawImage(hb, pad, 60); c.drawImage(ha, hb.width + 2 * pad, 60);
  c.font = "15px Segoe UI, sans-serif"; c.fillStyle = "#9aa4bf";
  c.fillText("Vorher (Basismodell)", pad, 60 + hb.height + 22);
  c.fillText("Nachher (personalisiert)", hb.width + 2 * pad, 60 + hb.height + 22);
  c.drawImage(tl, pad, 60 + hb.height + 40, w - 2 * pad, tlh);
  const pc = $("position-chart"), pch = Math.round((pc.height / pc.width) * (w - 2 * pad));
  const out2 = document.createElement("canvas");
  out2.width = out.width; out2.height = out.height + pch + pad;
  const c2 = out2.getContext("2d");
  c2.fillStyle = "#0b1020"; c2.fillRect(0, 0, out2.width, out2.height);
  c2.drawImage(out, 0, 0);
  c2.drawImage(pc, pad, out.height, w - 2 * pad, pch);
  download("blick_auswertung.png", out2.toDataURL("image/png"));
  return;
  // eslint-disable-next-line no-unreachable
  download("blick_auswertung.png", out.toDataURL("image/png"));
}

function exportCsv() {
  const lines = ["t_s,x_nachher,y_nachher,x_vorher,y_vorher,bereich_nachher"];
  for (const p of state.track) {
    lines.push([p.t.toFixed(3), p.px.toFixed(4), p.py.toFixed(4), p.bx.toFixed(4), p.by.toFixed(4),
      `"${REGIONS[regionOf(p.px, p.py)]}"`].join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  download("blick_daten.csv", URL.createObjectURL(blob));
}

// ---------------------------------------------------------------- Knöpfe
$("btn-start").onclick = () => runCalibration();
$("btn-recal").onclick = () => { state.track = []; runCalibration(); };
$("btn-track").onclick = () => { state.track = []; runTracking(); };
$("btn-again").onclick = () => runTracking();
$("btn-new").onclick = () => { state.track = []; runCalibration(); };
$("btn-png").onclick = exportPng;
$("btn-csv").onclick = exportCsv;
$("btn-csv-time").onclick = () => {
  const lines = ["von_s,bis_s,bereich,feld,x_prozent_breite,y_prozent_hoehe,anteil_im_bereich_prozent,messpunkte"];
  for (const r of state.timeRows || []) {
    lines.push([r.from.toFixed(1), r.to.toFixed(1), `"${r.region}"`, r.cell, (100 * r.x).toFixed(1),
      (100 * r.y).toFixed(1), r.share.toFixed(0), r.n].join(","));
  }
  download("blick_zeittabelle.csv", URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" })));
};
window.addEventListener("beforeunload", () => {
  if (state.sid) navigator.sendBeacon("/api/end", new Blob([JSON.stringify({ sid: state.sid })],
    { type: "application/json" }));
});

/* Kamera prüfen: liefert sie ein helles Bild? Wenn nicht, automatisch die nächste
 * Kamera probieren (Infrarot-Kameras wie "Windows Hello"/"IR" zuletzt). */
async function findWorkingCamera() {
  await sleep(800);
  await grabFrame();
  if (lastBrightness >= 4) return true;
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const isIR = (d) => /\b(ir|infrared|hello)\b/i.test(d.label);
  devices.sort((a, b) => isIR(a) - isIR(b));
  for (const d of devices) {
    try { await startCamera(d.deviceId); } catch (e) { continue; }
    await sleep(1000);
    await grabFrame();
    if (lastBrightness >= 4) return true;
  }
  return false;
}

/** Anzeige unter der Vorschau: ist das Kamerabild brauchbar? */
async function updateCamStatus() {
  if (!$("start").classList.contains("hidden")) {
    await grabFrame();
    const ok = lastBrightness >= 4;
    $("cam-status").innerHTML = ok
      ? `✓ Kamerabild OK (Helligkeit ${lastBrightness.toFixed(0)}) – ${camLabel}`
      : `<b style="color:#ff6b6b">⚠ Kamerabild ist schwarz</b> – andere Kamera wählen oder Licht einschalten (${camLabel})`;
    $("btn-start").disabled = !ok;
  }
  setTimeout(updateCamStatus, 1000);
}

// Kameraauswahl (wichtig bei Geräten mit Front-, Rück- und IR-Kamera, z.B. Surface)
async function fillCameraList() {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("cam-select");
  const current = cam.srcObject ? cam.srcObject.getVideoTracks()[0].getSettings().deviceId : "";
  sel.innerHTML = devices.map((d, i) =>
    `<option value="${d.deviceId}" ${d.deviceId === current ? "selected" : ""}>${d.label || "Kamera " + (i + 1)}</option>`).join("");
  sel.onchange = async () => {
    try { await startCamera(sel.value); } catch (e) { alert(`Kamera konnte nicht gestartet werden: ${e.message}`); }
  };
}

// ---------------------------------------------------------------- Start
(async function init() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    $("server-status").textContent = h.ok ? "✓ Server bereit" : "Server antwortet nicht";
  } catch (e) {
    $("server-status").textContent = "✗ Server nicht erreichbar";
  }
  try {
    await startCamera();
    $("cam-msg").classList.add("hidden");
    if (!(await findWorkingCamera())) {
      $("cam-msg").classList.remove("hidden");
      $("cam-msg").textContent = "Keine Kamera liefert ein Bild (alle schwarz). Kamera-Abdeckung, " +
        "Windows-Datenschutzeinstellungen (Kamera) und andere Programme prüfen.";
    }
    await fillCameraList();
    updateCamStatus();
  } catch (e) {
    $("cam-msg").textContent = `Kamera nicht verfügbar: ${e.message}. Bitte Zugriff erlauben und die Seite neu laden.`;
  }
})();
