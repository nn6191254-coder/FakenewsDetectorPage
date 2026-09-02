// DOM Elements
const articleInput = document.getElementById('articleInput');
const wordCount = document.getElementById('wordCount');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const sampleBtn = document.getElementById('sampleBtn');
const clearBtn = document.getElementById('clearBtn');
const copyResultBtn = document.getElementById('copyResultBtn');
const signalList = document.getElementById('signalList');
const metricAccuracy = document.getElementById('metricAccuracy');
const metricPrecision = document.getElementById('metricPrecision');
const metricRecall = document.getElementById('metricRecall');
const metricF1 = document.getElementById('metricF1');
const modelSampleCount = document.getElementById('modelSampleCount');
const historyList = document.getElementById('historyList');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const historyCount = document.getElementById('historyCount');
const navButtons = document.querySelectorAll('.nav-btn');
const panels = document.querySelectorAll('.panel');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultContent = document.getElementById('resultContent');

// Iconic HUD Elements
const hudScoreWrapper = document.getElementById('hudScoreWrapper');
const hudAmbientGlow = document.getElementById('hudAmbientGlow');
const hudProgressRing = document.getElementById('hudProgressRing');
const hudStatusIcon = document.getElementById('hudStatusIcon');
const hudIconBadge = document.getElementById('hudIconBadge');
const hudVerdictPill = document.getElementById('hudVerdictPill');
const scorePercentage = document.getElementById('scorePercentage');
const scoreLabelText = document.getElementById('scoreLabelText');
const scoreSummaryText = document.getElementById('scoreSummaryText');
const hudSummaryCard = document.getElementById('hudSummaryCard');
const hudSummaryBadge = document.getElementById('hudSummaryBadge');
const hudConfidenceTag = document.getElementById('hudConfidenceTag');
const reliablePatternBar = document.getElementById('reliablePatternBar');
const misleadingPatternBar = document.getElementById('misleadingPatternBar');
const reliablePatternValue = document.getElementById('reliablePatternValue');
const misleadingPatternValue = document.getElementById('misleadingPatternValue');
const reliableSubtext = document.getElementById('reliableSubtext');
const misleadingSubtext = document.getElementById('misleadingSubtext');

let sampleArticles = [];
let currentAnalysisData = null;
let scoreAnimationTimer = null;

// ==========================================
// Initialization & Startup
// ==========================================
document.addEventListener('DOMContentLoaded', async () => {
  updateTextStats();
  await loadModelInfo();
  await loadSamples();
  await loadHistory();
  initEventListeners();
});

// ==========================================
// Event Listeners Initialization
// ==========================================
function initEventListeners() {
  if (articleInput) {
    articleInput.addEventListener('input', updateTextStats);
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', analyzeArticle);
  }

  if (sampleBtn) {
    sampleBtn.addEventListener('click', loadRandomSample);
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', clearInput);
  }

  if (copyResultBtn) {
    copyResultBtn.addEventListener('click', copyCurrentAnalysis);
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener('click', clearAllHistory);
  }

  // Navigation Panel Switching
  navButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-target');
      setPanel(target);
    });
  });

  // Sample pill buttons
  document.querySelectorAll('.sample-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const idx = parseInt(pill.getAttribute('data-sample-idx'), 10);
      if (sampleArticles && sampleArticles[idx]) {
        articleInput.value = sampleArticles[idx].text;
        updateTextStats();
        setPanel('analyzer');
        showNotification(`Loaded sample: ${sampleArticles[idx].title}`, 'success');
      }
    });
  });

  // Keyboard Shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      analyzeArticle();
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 'l' && document.activeElement === articleInput) {
      e.preventDefault();
      clearInput();
    }
  });
}

// ==========================================
// Panel Navigation
// ==========================================
function setPanel(targetId) {
  navButtons.forEach((btn) => {
    btn.classList.toggle('active', btn.getAttribute('data-target') === targetId);
  });

  panels.forEach((panel) => {
    panel.classList.toggle('active-panel', panel.id === targetId);
  });
}

// ==========================================
// Word and Character Counting
// ==========================================
function updateTextStats() {
  const text = articleInput.value.trim();
  const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
  const chars = articleInput.value.length;

  if (wordCount) wordCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
  if (charCount) charCount.textContent = `${chars} char${chars === 1 ? '' : 's'}`;
}

// ==========================================
// Fetch Model Info
// ==========================================
async function loadModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (!res.ok) return;
    const data = await res.json();

    if (data.metrics) {
      if (metricAccuracy) metricAccuracy.textContent = Number(data.metrics.accuracy).toFixed(3);
      if (metricPrecision) metricPrecision.textContent = Number(data.metrics.precision).toFixed(3);
      if (metricRecall) metricRecall.textContent = Number(data.metrics.recall).toFixed(3);
      if (metricF1) metricF1.textContent = Number(data.metrics.f1).toFixed(3);
    }
    if (data.total_samples && modelSampleCount) {
      modelSampleCount.textContent = `${data.total_samples} verified cases`;
    }
  } catch (err) {
    console.warn('Could not load initial model info:', err);
  }
}

// ==========================================
// Fetch Samples
// ==========================================
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    if (!res.ok) return;
    sampleArticles = await res.json();
  } catch (err) {
    console.warn('Could not load samples:', err);
  }
}

function loadRandomSample() {
  if (!sampleArticles || sampleArticles.length === 0) return;
  const randIdx = Math.floor(Math.random() * sampleArticles.length);
  const sample = sampleArticles[randIdx];
  articleInput.value = sample.text;
  updateTextStats();
  showNotification(`Loaded: ${sample.title} (${sample.category})`, 'success');
}

// ==========================================
// Analyze Article Action
// ==========================================
async function analyzeArticle() {
  const text = articleInput.value.trim();

  if (!text) {
    showNotification('Please enter or paste an article to analyze.', 'error');
    articleInput.focus();
    return;
  }

  if (text.length < 10) {
    showNotification('Text is too short. Please provide at least 10 characters.', 'error');
    return;
  }

  // Show Loading UI
  if (loadingSpinner) loadingSpinner.style.display = 'flex';
  if (resultContent) resultContent.style.display = 'none';
  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-small"></span> Quantum Scanning...';
  }

  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Failed to analyze text.');
    }

    currentAnalysisData = data;
    updatePredictionDisplay(data);
    await loadHistory();
    showNotification('Intelligence analysis complete!', 'success');
  } catch (error) {
    showNotification(error.message || 'Error occurred during analysis.', 'error');
  } finally {
    if (loadingSpinner) loadingSpinner.style.display = 'none';
    if (resultContent) resultContent.style.display = 'block';
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> Analyze Article';
    }
  }
}

// ==========================================
// Animate Score Counter (Rolling Odometer)
// ==========================================
function animateScoreCounter(targetScore, duration = 850) {
  if (!scorePercentage) return;
  if (scoreAnimationTimer) cancelAnimationFrame(scoreAnimationTimer);

  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1.0);
    // Smooth easeOutExpo
    const easeProgress = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);
    const currentVal = (targetScore * easeProgress).toFixed(1);

    scorePercentage.textContent = `${currentVal}%`;

    if (progress < 1.0) {
      scoreAnimationTimer = requestAnimationFrame(update);
    } else {
      scorePercentage.textContent = `${targetScore.toFixed(1)}%`;
    }
  }

  scoreAnimationTimer = requestAnimationFrame(update);
}

// ==========================================
// Update Result Display (Iconic HUD)
// ==========================================
function updatePredictionDisplay(result) {
  const isReliable = result.label === 'Reliable';
  const reliableScore = Number(result.reliable_score ?? 50);
  const misleadingScore = Number(result.misleading_score ?? 50);
  const mainScore = isReliable ? reliableScore : misleadingScore;

  // 1. Update HUD Wrapper State & Ambient Glow
  if (hudScoreWrapper) {
    hudScoreWrapper.className = `hud-score-wrapper ${isReliable ? 'is-reliable' : 'is-misleading'}`;
  }

  // 2. Animate Score Number
  animateScoreCounter(mainScore);

  // 3. Update Status Icon & Verdict Badges
  if (hudStatusIcon) {
    hudStatusIcon.textContent = isReliable ? '🛡️' : '⚠️';
  }

  if (scoreLabelText) {
    scoreLabelText.textContent = isReliable ? 'AUTHENTIC / VERIFIED' : 'MISLEADING / HIGH RISK';
  }

  if (hudSummaryBadge) {
    hudSummaryBadge.textContent = isReliable ? 'VERIFIED CREDIBILITY' : 'ANOMALY DETECTED';
  }

  if (hudConfidenceTag) {
    hudConfidenceTag.textContent = `${result.confidence}% ${isReliable ? 'Confidence' : 'Risk Index'}`;
  }

  if (scoreSummaryText) {
    if (isReliable) {
      scoreSummaryText.textContent = `High authenticity alignment (${reliableScore}%). Text exhibits standard journalistic source citations, balanced sentiment, and verified structural attribution.`;
    } else {
      scoreSummaryText.textContent = `Elevated deception threat (${misleadingScore}%). Detected suspicious patterns such as sensational clickbait, emotional distortion, unverified conspiracies, or financial scam markers.`;
    }
  }

  if (reliableSubtext) {
    reliableSubtext.textContent = isReliable ? 'Factual Journalistic Rigor' : 'Weak Attribution Cues';
  }

  if (misleadingSubtext) {
    misleadingSubtext.textContent = isReliable ? 'Minimal Deception Threat' : 'Critical Red Flags Found';
  }

  // 4. Circular Ring Animation (r=88 => circumference = 2 * PI * 88 ≈ 552.92)
  if (hudProgressRing) {
    const circumference = 2 * Math.PI * 88;
    const offset = circumference - (mainScore / 100) * circumference;
    hudProgressRing.style.strokeDasharray = `${circumference - offset}, ${circumference}`;
    hudProgressRing.style.stroke = isReliable
      ? 'url(#hudGradientReliable)'
      : 'url(#hudGradientMisleading)';
    hudProgressRing.style.filter = isReliable
      ? 'url(#neonGlowReliable)'
      : 'url(#neonGlowMisleading)';
  }

  // 5. Pattern Bars & Numbers
  if (reliablePatternBar) reliablePatternBar.style.width = `${reliableScore}%`;
  if (misleadingPatternBar) misleadingPatternBar.style.width = `${misleadingScore}%`;
  if (reliablePatternValue) reliablePatternValue.textContent = `${reliableScore}%`;
  if (misleadingPatternValue) misleadingPatternValue.textContent = `${misleadingScore}%`;

  // 6. Render Signals
  renderSignals(result.signals || []);

  // 7. Update Metrics if returned
  if (result.metrics) {
    if (metricAccuracy) metricAccuracy.textContent = Number(result.metrics.accuracy).toFixed(3);
    if (metricPrecision) metricPrecision.textContent = Number(result.metrics.precision).toFixed(3);
    if (metricRecall) metricRecall.textContent = Number(result.metrics.recall).toFixed(3);
    if (metricF1) metricF1.textContent = Number(result.metrics.f1).toFixed(3);
  }

  // 8. Show Copy button
  if (copyResultBtn) copyResultBtn.style.display = 'inline-flex';
}

// ==========================================
// Render Signal Items
// ==========================================
function renderSignals(signals) {
  if (!signalList) return;

  if (!signals || signals.length === 0) {
    signalList.innerHTML = '<li class="signal-item"><p class="signal-desc">No signals evaluated.</p></li>';
    return;
  }

  signalList.innerHTML = signals
    .map((sig) => {
      let pillClass = 'pill-neutral';
      if (sig.severity === 'safe') pillClass = 'pill-safe';
      else if (sig.severity === 'warning') pillClass = 'pill-warning';
      else if (sig.severity === 'danger') pillClass = 'pill-danger';

      return `
        <li class="signal-item ${sig.severity || 'neutral'}">
          <div class="signal-item-head">
            <span class="signal-name">${escapeHtml(sig.name)}</span>
            <span class="signal-pill ${pillClass}">${escapeHtml(sig.status)}</span>
          </div>
          <p class="signal-desc">${escapeHtml(sig.detail)}</p>
        </li>
      `;
    })
    .join('');
}

// ==========================================
// Clear Input & Reset Results
// ==========================================
function clearInput() {
  articleInput.value = '';
  updateTextStats();
  currentAnalysisData = null;

  if (hudScoreWrapper) {
    hudScoreWrapper.className = 'hud-score-wrapper';
  }

  if (scorePercentage) {
    scorePercentage.textContent = '--%';
  }
  if (hudStatusIcon) {
    hudStatusIcon.textContent = '⚡';
  }
  if (scoreLabelText) {
    scoreLabelText.textContent = 'AWAITING INPUT';
  }
  if (hudSummaryBadge) {
    hudSummaryBadge.textContent = 'INTELLIGENCE REPORT';
  }
  if (hudConfidenceTag) {
    hudConfidenceTag.textContent = 'Standby';
  }
  if (scoreSummaryText) {
    scoreSummaryText.textContent = 'Enter text above or click a sample to inspect news credibility, factual attribution, and misleading cues.';
  }
  if (hudProgressRing) {
    hudProgressRing.style.strokeDasharray = '0, 553';
    hudProgressRing.style.stroke = 'url(#hudGradientReliable)';
    hudProgressRing.style.filter = 'none';
  }

  if (reliablePatternBar) reliablePatternBar.style.width = '0%';
  if (misleadingPatternBar) misleadingPatternBar.style.width = '0%';
  if (reliablePatternValue) reliablePatternValue.textContent = '--%';
  if (misleadingPatternValue) misleadingPatternValue.textContent = '--%';
  if (reliableSubtext) reliableSubtext.textContent = 'Factual Alignment';
  if (misleadingSubtext) misleadingSubtext.textContent = 'Manipulative Threat';

  if (copyResultBtn) copyResultBtn.style.display = 'none';

  if (signalList) {
    signalList.innerHTML = `
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Source Attribution</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">References to peer-reviewed journals, verified agencies, and official spokespersons.</p>
      </li>
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Clickbait & Sensationalism</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">Hyperbolic phrasing designed to trigger impulsive curiosity or outrage.</p>
      </li>
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Emotional Manipulation</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">Fear-inducing, alarmist, or strongly biased wording.</p>
      </li>
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Conspiracy Tropes</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">Suppressed truth tropes, secret cabal claims, and debunked theories.</p>
      </li>
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Spam / Fraud Signature</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">Lottery advance-fee scams, prize traps, phishing links, or urgency payment prompts.</p>
      </li>
      <li class="signal-item placeholder">
        <div class="signal-item-head">
          <span class="signal-name">Stylometry & Quality</span>
          <span class="signal-pill pill-neutral">Pending</span>
        </div>
        <p class="signal-desc">Excessive capitalization, exclamation mark clusters, and structural anomalies.</p>
      </li>
    `;
  }
}

// ==========================================
// Copy Analysis Result
// ==========================================
async function copyCurrentAnalysis() {
  if (!currentAnalysisData) return;

  const signalsText = (currentAnalysisData.signals || [])
    .map((s) => `• ${s.name} [${s.status}]: ${s.detail}`)
    .join('\n');

  const report = [
    `AUTHENTIQ AI | ANALYSIS REPORT`,
    `==============================`,
    `Verdict: ${currentAnalysisData.label} (${currentAnalysisData.confidence}% confidence)`,
    `Reliability Score: ${currentAnalysisData.reliable_score}%`,
    `Misleading/Risk Score: ${currentAnalysisData.misleading_score}%`,
    ``,
    `Linguistic Signals:`,
    signalsText,
    ``,
    `Article Excerpt:`,
    articleInput.value.slice(0, 300) + (articleInput.value.length > 300 ? '...' : ''),
  ].join('\n');

  try {
    await navigator.clipboard.writeText(report);
    if (copyResultBtn) {
      const origText = copyResultBtn.innerHTML;
      copyResultBtn.innerHTML = '✓ Copied!';
      setTimeout(() => {
        copyResultBtn.innerHTML = origText;
      }, 1800);
    }
    showNotification('Analysis report copied to clipboard!', 'success');
  } catch (err) {
    showNotification('Could not copy to clipboard.', 'error');
  }
}

// ==========================================
// Load & Render History
// ==========================================
async function loadHistory() {
  if (!historyList) return;

  try {
    const res = await fetch('/api/history');
    if (!res.ok) return;
    const data = await res.json();
    const items = data.items || [];

    if (historyCount) historyCount.textContent = items.length;

    if (items.length === 0) {
      historyList.innerHTML = '<div class="empty-state">No saved analyses yet. Run an analysis to see records here.</div>';
      return;
    }

    historyList.innerHTML = items
      .map((item) => {
        const isRel = item.label === 'Reliable';
        const dateStr = item.created_at ? new Date(item.created_at).toLocaleString() : 'Recent';

        return `
          <div class="history-entry ${isRel ? 'history-reliable' : 'history-misleading'}">
            <div class="history-entry-head">
              <div>
                <span class="history-badge ${isRel ? 'badge-reliable' : 'badge-misleading'}">
                  ${isRel ? '✓ Reliable' : '⚠️ Misleading'}
                </span>
                <span class="history-date">${escapeHtml(dateStr)}</span>
              </div>
              <strong class="history-conf">${Number(item.confidence).toFixed(1)}% Conf.</strong>
            </div>
            <p class="history-preview">${escapeHtml(item.article)}</p>
            <div class="history-actions-inline">
              <button class="history-action view-btn" data-id="${item.id}">🔍 View</button>
              <button class="history-action export-btn" data-export-id="${item.id}">📥 Export TXT</button>
              <button class="history-action delete-btn" data-delete-id="${item.id}">🗑️ Delete</button>
            </div>
          </div>
        `;
      })
      .join('');

    // Attach Action Listeners
    historyList.querySelectorAll('.view-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-id');
        try {
          const r = await fetch(`/api/history/${id}`);
          const item = await r.json();
          if (!r.ok || !item) return;

          articleInput.value = item.article;
          updateTextStats();
          currentAnalysisData = {
            label: item.label,
            confidence: item.confidence,
            reliability: item.reliability,
            reliable_score: Number((item.reliability * 100).toFixed(1)),
            misleading_score: Number(((1.0 - item.reliability) * 100).toFixed(1)),
            signals: item.signals,
            metrics: item.metrics,
          };
          updatePredictionDisplay(currentAnalysisData);
          setPanel('analyzer');
          showNotification('Loaded historic analysis record.', 'success');
        } catch (err) {
          showNotification('Failed to load historic record.', 'error');
        }
      });
    });

    historyList.querySelectorAll('.delete-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-delete-id');
        try {
          const r = await fetch(`/api/history/${id}`, { method: 'DELETE' });
          if (r.ok) {
            await loadHistory();
            showNotification('Record deleted.', 'success');
          }
        } catch (err) {
          showNotification('Failed to delete record.', 'error');
        }
      });
    });

    historyList.querySelectorAll('.export-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const id = btn.getAttribute('data-export-id');
        try {
          const r = await fetch(`/api/history/${id}`);
          const item = await r.json();
          if (!r.ok || !item) return;

          const signalsTxt = (item.signals || [])
            .map((s) => `• ${s.name} [${s.status}]: ${s.detail}`)
            .join('\n');

          const content = [
            `AUTHENTIQ AI ANALYSIS REPORT`,
            `===========================`,
            `Date: ${new Date(item.created_at).toLocaleString()}`,
            `Verdict: ${item.label}`,
            `Confidence: ${item.confidence}%`,
            `Reliability Score: ${Number(item.reliability * 100).toFixed(1)}%`,
            ``,
            `DETECTED SIGNALS:`,
            signalsTxt,
            ``,
            `ARTICLE CONTENT:`,
            item.article,
          ].join('\n');

          const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `authentiq-analysis-${id}.txt`;
          a.click();
          URL.revokeObjectURL(url);
          showNotification('Report file exported.', 'success');
        } catch (err) {
          showNotification('Failed to export record.', 'error');
        }
      });
    });
  } catch (err) {
    console.error('History load failed:', err);
  }
}

// ==========================================
// Clear All History
// ==========================================
async function clearAllHistory() {
  if (!confirm('Are you sure you want to delete all saved analyses?')) return;

  try {
    const res = await fetch('/api/history/clear', { method: 'DELETE' });
    if (res.ok) {
      await loadHistory();
      showNotification('History cleared successfully.', 'success');
    }
  } catch (err) {
    showNotification('Failed to clear history.', 'error');
  }
}

// ==========================================
// Toast Notification Utility
// ==========================================
function showNotification(message, type = 'success') {
  const existing = document.querySelectorAll('.notification');
  existing.forEach((el) => el.remove());

  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.innerHTML = `${type === 'error' ? '⚠️' : '✓'} <span>${escapeHtml(message)}</span>`;
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.classList.add('show');
  }, 20);

  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  }, 3500);
}

// ==========================================
// Utility: Escape HTML
// ==========================================
function escapeHtml(value) {
  if (!value) return '';
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
