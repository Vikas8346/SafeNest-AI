const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

const connDot = document.getElementById('connDot');
const connLabel = document.getElementById('connLabel');
const riskBanner = document.getElementById('riskBanner');
const riskLevel = document.getElementById('riskLevel');
const riskDetail = document.getElementById('riskDetail');
const inferLabel = document.getElementById('inferLabel');
const detList = document.getElementById('detList');
const detCount = document.getElementById('detCount');
const logList = document.getElementById('logList');
const scanLine = document.getElementById('scanLine');
const familyPhone = document.getElementById('familyPhone');
const captureBtn = document.getElementById('captureBtn');
const callBtn = document.getElementById('callBtn');
const momentStatus = document.getElementById('momentStatus');
const momentPreview = document.getElementById('momentPreview');

const SEND_INTERVAL_MS = 700;
const JPEG_QUALITY = 0.6;

const COLORS = {
  person: '#4FA3E3',
  danger: '#E6383D',
  neutral: '#5C6470',
};

let lastLoggedLevel = null;
let sending = false;
let currentFrameDataUrl = null;
let bestMoment = null;
let familyAlerted = false;
let savingAlert = false;
let cameraStream = null;
let loopTimer = null;
let cameraEnabled = true;

familyPhone.value = localStorage.getItem('safenest-family-phone') || '';
familyPhone.addEventListener('input', () => {
  localStorage.setItem('safenest-family-phone', familyPhone.value.trim());
});

function logEvent(text, level) {
  const row = document.createElement('div');
  row.className = 'log-row' + (level === 'CRITICAL' ? ' crit' : level === 'HIGH RISK' ? ' high' : '');
  const t = new Date().toLocaleTimeString();
  row.innerHTML = `<span class="t">${t}</span>${text}`;
  logList.appendChild(row);
  while (logList.children.length > 30) logList.removeChild(logList.firstChild);
}

function setConn(ok) {
  connDot.className = 'dot ' + (ok ? 'dot-on' : 'dot-off');
  connLabel.textContent = ok ? 'LIVE' : 'DISCONNECTED';
}

async function startCamera() {
  if (!cameraEnabled) return;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 }, audio: false });
    video.srcObject = cameraStream;
    await new Promise(res => { video.onloadedmetadata = res; });
    video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    cameraToggle.textContent = 'Camera off';
    cameraToggle.setAttribute('aria-pressed', 'true');
    logEvent('Camera stream started', null);
    setConn(true);
    loop();
  } catch (err) {
    connLabel.textContent = 'CAMERA ERROR';
    riskDetail.textContent = 'Could not access camera: ' + err.message;
    logEvent('Camera access failed: ' + err.message, 'CRITICAL');
  }
}

function clearDetectionState() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  detList.innerHTML = '<div class="det-empty">Camera is off</div>';
  detCount.textContent = '0';
  riskBanner.dataset.level = 'idle';
  riskLevel.textContent = 'STANDBY';
  riskDetail.textContent = 'Camera is off';
  inferLabel.textContent = '-- ms/frame';
  scanLine.className = 'scan-line';
  lastLoggedLevel = null;
}

function stopCamera() {
  cameraEnabled = false;
  if (loopTimer) clearTimeout(loopTimer);
  loopTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  video.srcObject = null;
  setConn(false);
  connLabel.textContent = 'CAMERA OFF';
  clearDetectionState();
  logEvent('Camera stream stopped', null);
}

function toggleCamera() {
  if (cameraEnabled) {
    stopCamera();
  } else {
    cameraEnabled = true;
    cameraToggle.textContent = 'Camera off';
    cameraToggle.setAttribute('aria-pressed', 'true');
    connLabel.textContent = 'INITIALIZING';
    startCamera();
  }
  cameraToggle.textContent = cameraEnabled ? 'Camera off' : 'Camera on';
  cameraToggle.setAttribute('aria-pressed', String(cameraEnabled));
}

function drawCornerBox(x1, y1, x2, y2, color) {
  const w = x2 - x1, h = y2 - y1;
  const cl = Math.max(10, Math.min(w, h) * 0.22); // corner length
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  // top-left
  ctx.moveTo(x1, y1 + cl); ctx.lineTo(x1, y1); ctx.lineTo(x1 + cl, y1);
  // top-right
  ctx.moveTo(x2 - cl, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + cl);
  // bottom-right
  ctx.moveTo(x2, y2 - cl); ctx.lineTo(x2, y2); ctx.lineTo(x2 - cl, y2);
  // bottom-left
  ctx.moveTo(x1 + cl, y2); ctx.lineTo(x1, y2); ctx.lineTo(x1, y2 - cl);
  ctx.stroke();
}

function drawLabel(text, x, y, color) {
  ctx.font = '600 12px JetBrains Mono, monospace';
  const padX = 5;
  const tw = ctx.measureText(text).width;
  const labelHeight = 16;
  const drawX = Math.max(0, Math.min(x, overlay.width - tw - padX * 2));
  const drawY = y >= labelHeight ? y - labelHeight : y;
  const textY = y >= labelHeight ? y - 4 : y + labelHeight - 4;
  ctx.fillStyle = color;
  ctx.fillRect(drawX, drawY, tw + padX * 2, labelHeight);
  ctx.fillStyle = '#0B0E11';
  ctx.fillText(text, drawX + padX, textY);
}

function render(detections) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  detList.innerHTML = '';

  if (detections.length === 0) {
    detList.innerHTML = '<div class="det-empty">No objects in frame</div>';
  }

  detections.forEach(d => {
    const [nx1, ny1, nx2, ny2] = d.box;
    // Map horizontal coordinates to match the CSS-mirrored video feed
    const x1 = (1 - nx2) * overlay.width;
    const x2 = (1 - nx1) * overlay.width;
    const y1 = ny1 * overlay.height;
    const y2 = ny2 * overlay.height;
    const color = COLORS[d.category] || COLORS.neutral;

    drawCornerBox(x1, y1, x2, y2, color);
    drawLabel(`${d.class} ${(d.conf * 100).toFixed(0)}%`, x1, y1, color);

    const row = document.createElement('div');
    row.className = 'det-row ' + d.category;
    row.innerHTML = `<span>${d.class}</span><span class="conf">${(d.conf * 100).toFixed(0)}%</span>`;
    detList.appendChild(row);
  });

  detCount.textContent = detections.length;
}

function saveMoment() {
  if (!bestMoment) return;
  const link = document.createElement('a');
  link.href = bestMoment.dataUrl;
  link.download = `safenest-moment-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
  link.click();
  logEvent('Alert moment saved to this device', null);
}

function callFamily() {
  return triggerFamilyCall({ label: 'CRITICAL', score: 70, object: 'manual emergency alert' });
}

async function triggerFamilyCall(risk) {
  momentStatus.textContent = 'Calling family member...';
  try {
    const response = await fetch('/call-family', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ risk }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'call failed');
    if (result.simulated) {
      momentStatus.textContent = `Dev Simulation: Call active (${result.call_sid})`;
      logEvent(`[TEST MODE] Emergency call simulated (${result.call_sid})`, 'CRITICAL');
    } else {
      momentStatus.textContent = `Family called via Twilio (${result.call_sid})`;
      logEvent(`Emergency call placed through Twilio (${result.call_sid})`, 'CRITICAL');
    }
  } catch (error) {
    momentStatus.textContent = 'Family call failed: see event log';
    logEvent('Emergency call failed: ' + error.message, 'CRITICAL');
  }
}

let alertCooldownTimer = null;

function speakAlert(objectName) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const text = objectName
    ? `Emergency alert. ${objectName} detected. Papa, please help.`
    : 'Emergency alert. Papa, please help.';
  const message = new SpeechSynthesisUtterance(text);
  message.rate = 0.9;
  window.speechSynthesis.speak(message);
}

async function saveAlertToServer(risk, dataUrl) {
  if (savingAlert || !dataUrl) return;
  savingAlert = true;
  momentStatus.textContent = 'Saving alert image and data...';
  try {
    const response = await fetch('/save-alert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl, risk }),
    });
    if (!response.ok) throw new Error('save failed');
    const result = await response.json();
    momentStatus.textContent = `Saved safely: ${result.image}`;
    logEvent('Alert image and risk data saved on server', risk.label);
  } catch (error) {
    momentStatus.textContent = 'Could not save alert data';
    logEvent('Alert save failed: ' + error.message, 'CRITICAL');
  } finally {
    savingAlert = false;
  }
}

function captureBestMoment(risk, dataUrl) {
  if (!dataUrl || risk.score < 45 || (bestMoment && risk.score <= bestMoment.score)) return;
  bestMoment = { score: risk.score, dataUrl };
  captureBtn.disabled = false;
  momentPreview.src = dataUrl;
  momentPreview.hidden = false;
  momentStatus.textContent = `Best moment: ${risk.label} (${risk.score}/100)`;
  logEvent(`Best alert moment captured — ${risk.object || 'object'}`, risk.label);
}

async function maybeAlertFamily(risk) {
  if (familyAlerted || risk.label !== 'CRITICAL') return;
  familyAlerted = true;
  speakAlert(risk.object);
  await saveAlertToServer(risk, currentFrameDataUrl);
  await triggerFamilyCall(risk);

  // Allow re-triggering after a 25-second cooldown if danger persists
  if (alertCooldownTimer) clearTimeout(alertCooldownTimer);
  alertCooldownTimer = setTimeout(() => {
    familyAlerted = false;
  }, 25000);
}

function applyRisk(risk, dataUrl) {
  const levelMap = {
    'SAFE': { key: 'safe', cls: '' },
    'WARNING': { key: 'warning', cls: '' },
    'HIGH RISK': { key: 'high', cls: 'active' },
    'CRITICAL': { key: 'critical', cls: 'active' },
  };
  const info = levelMap[risk.label] || { key: 'idle', cls: '' };
  riskBanner.dataset.level = info.key;
  riskLevel.textContent = risk.label;
  scanLine.className = 'scan-line ' + info.cls;

  if (risk.object) {
    if (risk.has_person) {
      riskDetail.textContent = `${risk.object} near person · score ${risk.score}/100`;
    } else {
      riskDetail.textContent = `Hazard detected: ${risk.object} in monitored area · score ${risk.score}/100`;
    }
  } else {
    riskDetail.textContent = 'No dangerous hazards detected in frame';
  }

  if ((risk.label === 'HIGH RISK' || risk.label === 'CRITICAL') && risk.label !== lastLoggedLevel) {
    const desc = risk.has_person
      ? `${risk.label} — ${risk.object || 'hazard'} near person (score ${risk.score})`
      : `${risk.label} — ${risk.object || 'hazard'} detected (score ${risk.score})`;
    logEvent(desc, risk.label);
  }
  captureBestMoment(risk, dataUrl);
  maybeAlertFamily(risk);
  if (risk.label === 'SAFE' && lastLoggedLevel && lastLoggedLevel !== 'SAFE') {
    logEvent('Risk cleared — back to SAFE', null);
    familyAlerted = false;
    if (alertCooldownTimer) clearTimeout(alertCooldownTimer);
  }
  lastLoggedLevel = risk.label;
}

async function sendFrame() {
  if (!cameraEnabled || sending) return;
  sending = true;
  try {
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
    currentFrameDataUrl = dataUrl;

    const res = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!res.ok) throw new Error('server error ' + res.status);
    const data = await res.json();

    setConn(true);
    inferLabel.textContent = `${data.infer_ms} ms/frame`;
    render(data.detections);
    applyRisk(data.risk, dataUrl);
  } catch (err) {
    setConn(false);
    connLabel.textContent = 'CONN ERROR';
  } finally {
    sending = false;
  }
}

function loop() {
  if (!cameraEnabled) return;
  sendFrame();
  loopTimer = setTimeout(loop, SEND_INTERVAL_MS);
}

const cameraToggle = document.getElementById('cameraToggle');
cameraToggle.addEventListener('click', toggleCamera);
startCamera();
captureBtn.addEventListener('click', saveMoment);
callBtn.addEventListener('click', callFamily);
