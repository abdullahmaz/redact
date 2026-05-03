// Redact dashboard frontend.

const state = {
  classes: [],
  baseline: [],
  current: [],
  history: [],
  selected: new Set(),
  lastForget: [],
};

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
}

async function loadStatus() {
  const s = await fetchJSON("/api/status");
  state.classes = s.classes;
  state.baseline = s.baseline_per_class;
  state.current = s.current_per_class;
  state.history = s.history;
  document.getElementById("device-pill").textContent = "device: " + s.device;
  document.getElementById("ckpt-pill").textContent = "checkpoint: " + s.checkpoint;
  renderAll();
}

function renderClasses() {
  const grid = document.getElementById("class-grid");
  grid.innerHTML = "";
  state.classes.forEach((name, c) => {
    const cell = document.createElement("div");
    cell.className = "class-cell" + (state.selected.has(c) ? " selected" : "");
    cell.dataset.class = c;
    cell.innerHTML = `<span class="name">${c}: ${name}</span><span class="acc">${fmtPct(state.current[c] || 0)}</span>`;
    cell.onclick = () => {
      if (state.selected.has(c)) state.selected.delete(c);
      else state.selected.add(c);
      renderClasses();
    };
    grid.appendChild(cell);
  });
}

function renderChart() {
  const chart = document.getElementById("acc-chart");
  chart.innerHTML = "";
  state.classes.forEach((name, c) => {
    const base = state.baseline[c] || 0;
    const cur = state.current[c] || 0;
    const delta = cur - base;
    const isForget = state.lastForget.includes(c);
    const row = document.createElement("div");
    row.className = "acc-row";
    row.innerHTML = `
      <div class="name">${c}: ${name}</div>
      <div class="bar"><div class="fill" style="width:${(base * 100).toFixed(1)}%"></div></div>
      <div class="bar ${isForget ? "forget" : ""}"><div class="fill current" style="width:${(cur * 100).toFixed(1)}%"></div></div>
      <div class="delta">${(delta >= 0 ? "+" : "") + (delta * 100).toFixed(1)}%</div>
    `;
    chart.appendChild(row);
  });
}

function renderMetrics() {
  const m = document.getElementById("metrics");
  const baselineMean = state.baseline.reduce((a, b) => a + b, 0) / 10;
  const currentMean = state.current.reduce((a, b) => a + b, 0) / 10;
  let adf = "—", adr = "—";
  if (state.history.length) {
    const last = state.history[state.history.length - 1];
    adf = fmtPct(last.ADf);
    adr = fmtPct(last.ADr);
  }
  m.innerHTML = `
    <div class="metric"><div class="label">baseline mean acc</div><div class="value">${fmtPct(baselineMean)}</div></div>
    <div class="metric"><div class="label">current mean acc</div><div class="value">${fmtPct(currentMean)}</div></div>
    <div class="metric"><div class="label">ADf (last run)</div><div class="value bad">${adf}</div></div>
    <div class="metric"><div class="label">ADr (last run)</div><div class="value good">${adr}</div></div>
  `;
}

function renderTimings() {
  const t = document.getElementById("timings");
  t.innerHTML = "";
  if (!state.history.length) {
    t.innerHTML = `<p class="muted">No runs yet.</p>`;
    return;
  }
  const last = state.history[state.history.length - 1];
  const total = last.timings.total || 1;
  ["noise", "impair", "repair"].forEach((p) => {
    const v = last.timings[p];
    const row = document.createElement("div");
    row.className = "timing-row";
    row.innerHTML = `
      <div>${p}</div>
      <div class="bar"><div class="fill" style="width:${((v / total) * 100).toFixed(1)}%"></div></div>
      <div>${v.toFixed(2)}s</div>
    `;
    t.appendChild(row);
  });
  const tot = document.createElement("div");
  tot.className = "timing-row";
  tot.innerHTML = `<div><strong>total</strong></div><div></div><div><strong>${last.timings.total.toFixed(2)}s</strong></div>`;
  t.appendChild(tot);
}

function renderHistory() {
  const tbody = document.querySelector("#history tbody");
  tbody.innerHTML = "";
  if (!state.history.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="muted">No runs yet.</td>`;
    tbody.appendChild(tr);
    return;
  }
  state.history.forEach((h, i) => {
    const tr = document.createElement("tr");
    const names = h.forget_classes.map((c) => state.classes[c]).join(", ");
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${names}</td>
      <td>${fmtPct(h.ADf)}</td>
      <td>${fmtPct(h.ADr)}</td>
      <td>${h.timings.noise.toFixed(2)}s</td>
      <td>${h.timings.impair.toFixed(2)}s</td>
      <td>${h.timings.repair.toFixed(2)}s</td>
      <td>${h.timings.total.toFixed(2)}s</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderSampleFilter() {
  const sel = document.getElementById("sample-filter");
  if (sel.options.length > 1) return;
  state.classes.forEach((name, c) => {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = `${c}: ${name}`;
    sel.appendChild(o);
  });
}

async function refreshSamples() {
  const filter = document.getElementById("sample-filter").value;
  const container = document.getElementById("samples");
  container.innerHTML = "<p class='muted'>Loading...</p>";

  const N = 10000;
  const targets = [];
  // pick 12 random indices, optionally filtered
  while (targets.length < 12) {
    const idx = Math.floor(Math.random() * N);
    if (filter === "all") targets.push(idx);
    else {
      try {
        const r = await fetchJSON(`/api/predict/${idx}`);
        if (r.true === Number(filter)) targets.push(idx);
      } catch (e) { /* skip */ }
      if (targets.length === 0 && targets.attemptCount > 200) break;
    }
  }

  container.innerHTML = "";
  for (const i of targets) {
    const r = await fetchJSON(`/api/predict/${i}`);
    const div = document.createElement("div");
    div.className = "sample";
    const correct = r.pred === r.true;
    const trueName = state.classes[r.true];
    const predName = state.classes[r.pred];
    div.innerHTML = `
      <img src="/api/sample/${i}" alt="" />
      <div class="info">
        <span class="true">true: ${trueName}</span>
        <span class="pred ${correct ? "correct" : "wrong"}">pred: ${predName}</span>
      </div>
    `;
    container.appendChild(div);
  }
}

async function runUnlearn() {
  if (state.selected.size === 0) {
    alert("Select at least one class to forget.");
    return;
  }
  const btn = document.getElementById("btn-unlearn");
  const status = document.getElementById("run-status");
  btn.disabled = true;
  status.textContent = "Running UNSIR... (noise → impair → repair)";

  const body = {
    forget_classes: Array.from(state.selected),
    noise_steps: Number(document.getElementById("cfg-noise-steps").value),
    noise_lambda: Number(document.getElementById("cfg-noise-lambda").value),
    impair_lr: Number(document.getElementById("cfg-impair-lr").value),
    repair_lr: Number(document.getElementById("cfg-repair-lr").value),
    per_class: Number(document.getElementById("cfg-per-class").value),
  };

  try {
    const t0 = performance.now();
    const r = await fetchJSON("/api/unlearn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    state.lastForget = body.forget_classes;
    status.textContent = `Done in ${dt}s — ADf=${fmtPct(r.result.ADf)}, ADr=${fmtPct(r.result.ADr)}`;
    await loadStatus();
    await refreshSamples();
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function reset() {
  const status = document.getElementById("run-status");
  status.textContent = "Resetting baseline...";
  await fetchJSON("/api/reset", { method: "POST" });
  state.lastForget = [];
  state.selected.clear();
  status.textContent = "Reset.";
  await loadStatus();
  await refreshSamples();
}

function renderAll() {
  renderClasses();
  renderChart();
  renderMetrics();
  renderTimings();
  renderHistory();
  renderSampleFilter();
}

document.getElementById("btn-unlearn").addEventListener("click", runUnlearn);
document.getElementById("btn-reset").addEventListener("click", reset);
document.getElementById("btn-refresh-samples").addEventListener("click", refreshSamples);
document.getElementById("sample-filter").addEventListener("change", refreshSamples);

(async () => {
  await loadStatus();
  await refreshSamples();
})();
