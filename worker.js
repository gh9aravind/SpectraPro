// worker.js — all heavy DSP runs here, off the main thread.
// Responsibilities:
//   1. STFT spectrogram (per-channel + combined "All")
//   2. Peak / RMS / Dynamic Range / clipping (per-channel + overall)
//   3. Spectral cutoff detection (fake-lossless / transcode heuristic)
//   4. Drawing the initial spectrogram view directly onto an OffscreenCanvas

const FFT_SIZE = 2048;
const HOP = 1024;

// ---------- FFT ----------
// Iterative in-place radix-2 Cooley-Tukey FFT (complex arrays, length must be pow2)
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curWr = 1, curWi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = i + k + half;
        const vr = re[b] * curWr - im[b] * curWi;
        const vi = re[b] * curWi + im[b] * curWr;
        const ur = re[a], ui = im[a];
        re[a] = ur + vr; im[a] = ui + vi;
        re[b] = ur - vr; im[b] = ui - vi;
        const nWr = curWr * wr - curWi * wi;
        const nWi = curWr * wi + curWi * wr;
        curWr = nWr; curWi = nWi;
      }
    }
  }
}

function hannWindow(size) {
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  return w;
}
const WINDOW = hannWindow(FFT_SIZE);

// STFT -> array of Float32Array(FFT_SIZE/2) linear magnitude, one per frame
function computeSTFT(samples, onProgress) {
  const half = FFT_SIZE / 2;
  const numFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP) + 1);
  const frames = new Array(numFrames);
  const re = new Float32Array(FFT_SIZE);
  const im = new Float32Array(FFT_SIZE);
  for (let f = 0; f < numFrames; f++) {
    const start = f * HOP;
    for (let i = 0; i < FFT_SIZE; i++) {
      const idx = start + i;
      re[i] = (idx < samples.length ? samples[idx] : 0) * WINDOW[i];
      im[i] = 0;
    }
    fft(re, im);
    const mags = new Float32Array(half);
    for (let k = 0; k < half; k++) mags[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / FFT_SIZE;
    frames[f] = mags;
    if (onProgress && (f & 127) === 0) onProgress(f / numFrames);
  }
  return frames;
}

function combineChannelsPower(frameArrays) {
  const numFrames = frameArrays[0].length;
  const half = frameArrays[0][0].length;
  const out = new Array(numFrames);
  for (let f = 0; f < numFrames; f++) {
    const combined = new Float32Array(half);
    for (let k = 0; k < half; k++) {
      let sumSq = 0;
      for (let c = 0; c < frameArrays.length; c++) {
        const v = frameArrays[c][f][k];
        sumSq += v * v;
      }
      combined[k] = Math.sqrt(sumSq / frameArrays.length);
    }
    out[f] = combined;
  }
  return out;
}

function magToDb(mag) { return 20 * Math.log10(Math.max(mag, 1e-9)); }

// Downsample frame list into a [cols x rows] dB matrix (flat, col-major) for display
function downsampleForDisplay(frames, cols, rows) {
  const numFrames = frames.length;
  const half = frames[0].length;
  const colStep = numFrames / cols;
  const rowStep = half / rows;
  const out = new Float32Array(cols * rows);
  for (let c = 0; c < cols; c++) {
    const fStart = Math.floor(c * colStep);
    const fEnd = Math.max(fStart + 1, Math.floor((c + 1) * colStep));
    for (let r = 0; r < rows; r++) {
      const kStart = Math.floor(r * rowStep);
      const kEnd = Math.max(kStart + 1, Math.floor((r + 1) * rowStep));
      let sumSq = 0, count = 0;
      for (let f = fStart; f < fEnd && f < numFrames; f++) {
        const frame = frames[f];
        for (let k = kStart; k < kEnd && k < half; k++) { sumSq += frame[k] * frame[k]; count++; }
      }
      out[c * rows + r] = magToDb(count > 0 ? Math.sqrt(sumSq / count) : 0);
    }
  }
  return out;
}

// ---------- color gradient (navy -> blue -> purple -> red -> orange -> yellow/white) ----------
const STOPS = [
  [0.00, [8, 8, 22]],
  [0.16, [32, 20, 95]],
  [0.36, [115, 25, 135]],
  [0.56, [205, 35, 70]],
  [0.74, [238, 105, 20]],
  [0.88, [251, 193, 45]],
  [1.00, [255, 251, 232]],
];
function dbToColor(db, minDb, maxDb) {
  let t = (db - minDb) / (maxDb - minDb);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [t0, c0] = STOPS[i], [t1, c1] = STOPS[i + 1];
    if (t >= t0 && t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return [c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f, c0[2] + (c1[2] - c0[2]) * f];
    }
  }
  return STOPS[STOPS.length - 1][1];
}

function drawSpectrogram(ctx, dbMatrix, cols, rows, minDb, maxDb) {
  const img = ctx.createImageData(cols, rows);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) {
      const db = dbMatrix[c * rows + r];
      const [rr, gg, bb] = dbToColor(db, minDb, maxDb);
      const y = rows - 1 - r; // low freq at bottom
      const idx = (y * cols + c) * 4;
      img.data[idx] = rr; img.data[idx + 1] = gg; img.data[idx + 2] = bb; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ---------- spectral cutoff / fake-lossless heuristic ----------
function detectSpectralCutoff(avgLinearSpectrum, sampleRate) {
  const half = avgLinearSpectrum.length;
  const binHz = (sampleRate / 2) / half;
  const dbs = new Float32Array(half);
  for (let k = 0; k < half; k++) dbs[k] = magToDb(avgLinearSpectrum[k]);
  let peakDb = -Infinity;
  for (let k = 0; k < half; k++) if (dbs[k] > peakDb) peakDb = dbs[k];

  const topStart = Math.floor(half * 0.9);
  const topSorted = Array.from(dbs.slice(topStart)).sort((a, b) => a - b);
  const noiseFloor = topSorted[Math.floor(topSorted.length / 2)] ?? -100;
  const threshold = Math.max(noiseFloor + 8, peakDb - 65);

  let cutoffBin = half - 1;
  for (let k = half - 1; k >= 1; k--) {
    if (dbs[k] > threshold && dbs[k - 1] > threshold) { cutoffBin = k; break; }
  }
  const cutoffHz = cutoffBin * binHz;

  const bandBins = Math.max(1, Math.round(1000 / binHz));
  const beforeIdx = Math.max(0, cutoffBin - Math.round(200 / binHz));
  const afterIdx = Math.min(half - 1, cutoffBin + bandBins);
  const dropPerKHz = dbs[beforeIdx] - dbs[afterIdx];
  const isSharp = dropPerKHz > 25;

  return { cutoffHz, nyquist: sampleRate / 2, isSharp, dropPerKHz };
}

// ---------- level stats ----------
function levelStats(samples) {
  let peak = 0, sumSq = 0, clipCount = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > peak) peak = a;
    sumSq += samples[i] * samples[i];
    if (a >= 0.999) clipCount++;
  }
  const rms = Math.sqrt(sumSq / samples.length);
  const peakDb = 20 * Math.log10(Math.max(peak, 1e-9));
  const rmsDb = 20 * Math.log10(Math.max(rms, 1e-9));
  return { peakDb, rmsDb, dr: peakDb - rmsDb, clipCount, peakLinear: peak, rmsLinear: rms };
}

// ---------- main ----------
// State kept inside the worker so channel toggles can redraw instantly
// without recomputing the STFT or shipping large arrays back and forth.
const DB_MIN = -100, DB_MAX = -5;
let state = { ctx: null, cols: 0, rows: 0, views: null };

self.onmessage = (e) => {
  const msg = e.data;

  if (msg.type === 'setView') {
    if (state.ctx && state.views && state.views[msg.view]) {
      drawSpectrogram(state.ctx, state.views[msg.view], state.cols, state.rows, DB_MIN, DB_MAX);
    }
    return;
  }

  // msg.type === 'analyze' (initial run)
  const { channels, sampleRate, canvas, targetCols, targetRows } = msg;
  const numCh = channels.length;

  const perChannelStats = channels.map(levelStats);
  let overallStats;
  {
    const peakLinear = Math.max(...perChannelStats.map((s) => s.peakLinear));
    const rmsLinear = Math.sqrt(perChannelStats.reduce((s, c) => s + c.rmsLinear * c.rmsLinear, 0) / numCh);
    const peakDb = 20 * Math.log10(Math.max(peakLinear, 1e-9));
    const rmsDb = 20 * Math.log10(Math.max(rmsLinear, 1e-9));
    const clipCount = Math.max(...perChannelStats.map((s) => s.clipCount));
    overallStats = { peakDb, rmsDb, dr: peakDb - rmsDb, clipCount };
  }

  const perChannelFrames = channels.map((ch, i) =>
    computeSTFT(ch, (p) => self.postMessage({ type: 'progress', value: (i + p) / numCh }))
  );
  const combinedFrames = numCh > 1 ? combineChannelsPower(perChannelFrames) : perChannelFrames[0];

  const half = FFT_SIZE / 2;
  const avgSpectrum = new Float32Array(half);
  for (let f = 0; f < combinedFrames.length; f++) {
    const fr = combinedFrames[f];
    for (let k = 0; k < half; k++) avgSpectrum[k] += fr[k];
  }
  for (let k = 0; k < half; k++) avgSpectrum[k] /= combinedFrames.length;
  const cutoff = detectSpectralCutoff(avgSpectrum, sampleRate);

  const views = { all: downsampleForDisplay(combinedFrames, targetCols, targetRows) };
  if (numCh > 1) {
    views.ch1 = downsampleForDisplay(perChannelFrames[0], targetCols, targetRows);
    views.ch2 = downsampleForDisplay(perChannelFrames[1], targetCols, targetRows);
  }

  state.cols = targetCols;
  state.rows = targetRows;
  state.views = views;
  if (canvas) {
    state.ctx = canvas.getContext('2d');
    drawSpectrogram(state.ctx, views.all, targetCols, targetRows, DB_MIN, DB_MAX);
  }

  // Only small, structured-clone-friendly data goes back to the main thread —
  // the (potentially large) dB matrices stay in the worker for redraws.
  self.postMessage({
    type: 'done',
    perChannelStats,
    overallStats,
    cutoff,
    hasCh2: numCh > 1,
    targetCols,
    targetRows,
    totalSamples: channels[0].length,
    fftSize: FFT_SIZE,
  });
};
