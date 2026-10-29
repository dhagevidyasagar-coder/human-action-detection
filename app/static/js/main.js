// HumanSense AI - Client Side Controller

let isWebcamRunning = false;
let videoStream = null;
let processingLoopId = null;
let lastFrameTime = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();

const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const hiddenCanvas = document.getElementById('hidden-canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');

const btnCamera = document.getElementById('btn-camera');
const btnCameraText = document.getElementById('btn-camera-text');
const btnDemoHappy = document.getElementById('btn-demo-happy');
const btnDemoFocused = document.getElementById('btn-demo-focused');
const btnDemoStanding = document.getElementById('btn-demo-standing');
const mediaUploader = document.getElementById('media-uploader');
const btnReset = document.getElementById('btn-reset');

const togglePose = document.getElementById('toggle-pose');
const toggleFace = document.getElementById('toggle-face');
const toggleMirror = document.getElementById('toggle-mirror');

// HUD Elements
const hudStateText = document.getElementById('hud-state-text');
const hudStateIndicator = document.getElementById('hud-state-indicator');
const hudAction = document.getElementById('hud-action');
const hudActionBar = document.getElementById('hud-action-bar');
const hudEmotion = document.getElementById('hud-emotion');
const hudEmotionBar = document.getElementById('hud-emotion-bar');
const alertBanner = document.getElementById('alert-banner');
const alertBannerText = document.getElementById('alert-banner-text');

// Insight Box Elements
const insightBox = document.getElementById('insight-box');
const insightTitle = document.getElementById('insight-title');
const insightDesc = document.getElementById('insight-desc');
const insightCategory = document.getElementById('insight-category');
const insightConf = document.getElementById('insight-conf');
const insightRecommendation = document.getElementById('insight-recommendation');

// Breakdown lists
const actionProbList = document.getElementById('action-prob-list');
const emotionProbList = document.getElementById('emotion-prob-list');
const actDominant = document.getElementById('act-dominant');
const emoDominant = document.getElementById('emo-dominant');
const actTopConf = document.getElementById('act-top-conf');
const emoTopConf = document.getElementById('emo-top-conf');

// Biometrics
const bioEar = document.getElementById('bio-ear');
const bioMar = document.getElementById('bio-mar');
const bioSmile = document.getElementById('bio-smile');
const bioBrow = document.getElementById('bio-brow');

// Metrics
const fpsCounter = document.getElementById('fps-counter');
const sessionTimer = document.getElementById('session-timer');
const scoreEngagement = document.getElementById('score-engagement');
const timelineContainer = document.getElementById('timeline-container');

// Toggle mirror view
toggleMirror.addEventListener('change', () => {
  if (toggleMirror.checked) {
    canvas.classList.add('mirrored');
    video.classList.add('mirrored');
  } else {
    canvas.classList.remove('mirrored');
    video.classList.remove('mirrored');
  }
});

// Setup mirror class initially
if (toggleMirror.checked) {
  canvas.classList.add('mirrored');
  video.classList.add('mirrored');
}

// Start / Stop Webcam
btnCamera.addEventListener('click', async () => {
  if (!isWebcamRunning) {
    await startWebcam();
  } else {
    stopWebcam();
  }
});

async function startWebcam() {
  try {
    document.getElementById('loading-overlay').classList.remove('hidden');
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = videoStream;
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      hiddenCanvas.width = canvas.width;
      hiddenCanvas.height = canvas.height;
      document.getElementById('loading-overlay').classList.add('hidden');
      isWebcamRunning = true;
      btnCameraText.textContent = 'Stop Webcam';
      btnCamera.classList.remove('btn-primary');
      btnCamera.classList.add('btn-secondary');
      runProcessingLoop();
    };
  } catch (err) {
    document.getElementById('loading-overlay').classList.add('hidden');
    console.warn('Webcam access error or permission denied:', err);
    alert('Could not access webcam. You can still test with the Demo buttons or upload an image/video.');
  }
}

function stopWebcam() {
  if (videoStream) {
    videoStream.getTracks().forEach(track => track.stop());
    videoStream = null;
  }
  isWebcamRunning = false;
  btnCameraText.textContent = 'Start Webcam';
  btnCamera.classList.add('btn-primary');
  btnCamera.classList.remove('btn-secondary');
  if (processingLoopId) {
    clearTimeout(processingLoopId);
    processingLoopId = null;
  }
}

// Frame capture & AI request loop
async function runProcessingLoop() {
  if (!isWebcamRunning) return;

  const now = performance.now();
  frameCount++;
  if (now - lastFpsUpdate >= 1000) {
    const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
    fpsCounter.textContent = `${fps} fps`;
    frameCount = 0;
    lastFpsUpdate = now;
  }

  // Draw video frame to hidden canvas
  hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
  const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.7);

  try {
    const response = await fetch('/api/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataUrl,
        draw_pose: togglePose.checked,
        draw_face: toggleFace.checked,
        return_annotated: true
      })
    });

    if (response.ok) {
      const data = await response.json();
      updateDashboard(data);

      if (data.annotated_image) {
        const img = new Image();
        img.onload = () => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        };
        img.src = data.annotated_image;
      }
    }
  } catch (err) {
    console.error('Frame processing error:', err);
  }

  if (isWebcamRunning) {
    processingLoopId = setTimeout(runProcessingLoop, 90); // ~11 requests/sec
  }
}

// Update UI with detection data
function updateDashboard(data) {
  const fused = data.fused_state;
  if (!fused) return;

  // 1. HUD State Pill
  hudStateText.textContent = fused.state_title;
  hudStateIndicator.style.background = fused.theme_color || '#3b82f6';
  hudStateIndicator.style.boxShadow = `0 0 10px ${fused.theme_color || '#3b82f6'}`;

  // 2. HUD Action & Emotion
  hudAction.textContent = data.action.label;
  const actConfPct = Math.round(data.action.confidence * 100);
  hudActionBar.style.width = `${actConfPct}%`;
  actTopConf.textContent = `${actConfPct}%`;
  actDominant.textContent = data.action.label;

  hudEmotion.textContent = data.emotion.label;
  const emoConfPct = Math.round(data.emotion.confidence * 100);
  hudEmotionBar.style.width = `${emoConfPct}%`;
  emoTopConf.textContent = `${emoConfPct}%`;
  emoDominant.textContent = data.emotion.label;

  // 3. Alert banner
  if (fused.category === 'Wellbeing Alert') {
    alertBanner.classList.remove('hidden');
    alertBannerText.textContent = '⚠️ Desk Slump / Drowsiness Alert! Take a break.';
  } else if (fused.category === 'Stress Alert') {
    alertBanner.classList.remove('hidden');
    alertBannerText.textContent = '⚠️ Mental Stress / Overwhelm Detected. Take deep breaths.';
  } else {
    alertBanner.classList.add('hidden');
  }

  // 4. Insight Box
  insightTitle.textContent = fused.state_title;
  insightDesc.textContent = fused.description;
  insightCategory.textContent = fused.category;
  insightCategory.style.borderColor = fused.theme_color;
  insightConf.textContent = `Confidence: ${Math.round(fused.fusion_confidence * 100)}%`;
  insightRecommendation.textContent = fused.recommendation;

  // 5. Modality Probability Bars
  renderProbList(actionProbList, data.action.probabilities, '#3b82f6');
  renderProbList(emotionProbList, data.emotion.probabilities, '#06b6d4');

  // 6. Biometrics
  if (data.emotion.biometrics) {
    bioEar.textContent = data.emotion.biometrics.ear ?? '--';
    bioMar.textContent = data.emotion.biometrics.mar ?? '--';
    bioSmile.textContent = data.emotion.biometrics.smile_index ?? '--';
    bioBrow.textContent = data.emotion.biometrics.brow_furrow ?? '--';
  }

  // 7. Stats & Timeline
  if (data.stats) {
    sessionTimer.textContent = data.stats.duration || '00:00';
    scoreEngagement.textContent = `${data.stats.engagement_score}%`;
    renderTimeline(data.stats.recent_timeline);
  }
}

function renderProbList(container, probs, accentColor) {
  if (!probs) return;
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 5);
  container.innerHTML = sorted.map(([label, val]) => {
    const pct = Math.round(val * 100);
    return `
      <div class="prob-row">
        <div class="prob-info">
          <span>${label}</span>
          <strong>${pct}%</strong>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: ${pct}%; background: ${accentColor};"></div>
        </div>
      </div>
    `;
  }).join('');
}

function renderTimeline(timeline) {
  if (!timeline || timeline.length === 0) return;
  timelineContainer.innerHTML = timeline.slice().reverse().map(item => `
    <div class="timeline-entry" style="border-left-color: ${item.color || '#3b82f6'};">
      <div>
        <span class="timeline-title">${item.title}</span>
        <div style="font-size: 11px; color: #9ca3af;">${item.action} • ${item.emotion}</div>
      </div>
      <div style="text-align: right;">
        <span class="timeline-time">${item.timestamp}</span>
        <div><span class="timeline-cat">${item.category || 'Normal'}</span></div>
      </div>
    </div>
  `).join('');
}

// Demo Sample Buttons
btnDemoHappy.addEventListener('click', async () => {
  stopWebcam();
  try {
    const res = await fetch('/api/demo_sample/happy');
    const data = await res.json();
    if (data.success) {
      updateDashboard(data);
      if (data.annotated_image) {
        displayStaticImage(data.annotated_image);
      }
    }
  } catch (e) {
    console.error('Error loading happy demo:', e);
  }
});

btnDemoFocused.addEventListener('click', async () => {
  stopWebcam();
  try {
    const res = await fetch('/api/demo_sample/focused');
    const data = await res.json();
    if (data.success) {
      updateDashboard(data);
      if (data.annotated_image) {
        displayStaticImage(data.annotated_image);
      }
    }
  } catch (e) {
    console.error('Error loading focused demo:', e);
  }
});

btnDemoStanding.addEventListener('click', async () => {
  stopWebcam();
  try {
    const res = await fetch('/api/demo_sample/standing');
    const data = await res.json();
    if (data.success) {
      updateDashboard(data);
      if (data.annotated_image) {
        displayStaticImage(data.annotated_image);
      }
    }
  } catch (e) {
    console.error('Error loading standing demo:', e);
  }
});

function displayStaticImage(dataUrl) {
  const img = new Image();
  img.onload = () => {
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = dataUrl;
}

// Media file uploader (Image/Video)
mediaUploader.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  stopWebcam();
  const reader = new FileReader();
  reader.onload = async (event) => {
    const dataUrl = event.target.result;
    try {
      const response = await fetch('/api/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          draw_pose: togglePose.checked,
          draw_face: toggleFace.checked,
          return_annotated: true
        })
      });
      const data = await response.json();
      if (data.success) {
        updateDashboard(data);
        if (data.annotated_image) {
          displayStaticImage(data.annotated_image);
        }
      }
    } catch (err) {
      console.error('Media upload processing failed:', err);
    }
  };
  reader.readAsDataURL(file);
});

// Session Reset
btnReset.addEventListener('click', async () => {
  await fetch('/api/reset_session', { method: 'POST' });
  timelineContainer.innerHTML = '<div class="empty-timeline">Session reset. Start camera or select a demo.</div>';
  scoreEngagement.textContent = '85%';
  sessionTimer.textContent = '00:00';
});
