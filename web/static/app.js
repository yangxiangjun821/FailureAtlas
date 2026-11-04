const LABELS = {
  rotation: "旋转",
  gaussian_noise: "高斯噪声",
  blur: "高斯模糊",
  occlusion: "中心遮挡",
  brightness: "亮度降低",
};

const STATIC_SHOWCASE = window.FAILUREATLAS_STATIC === true;

const UNIT = {
  rotation: "°",
  gaussian_noise: "",
  blur: "",
  occlusion: "",
  brightness: "",
};

const state = {
  data: null,
  projects: [],
  catalog: null,
  runs: [],
  system: null,
  selectedRuns: new Set(),
  selected: "rotation",
  poller: null,
  preflightTimer: null,
};
const $ = (selector) => document.querySelector(selector);
const percent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
const signedPercent = (value) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
const boundary = (value, name) => value == null ? "未触发" : `${value}${UNIT[name]}`;
const formatBytes = (value) => {
  if (value == null) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value, index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size.toFixed(index > 1 ? 1 : 0)} ${units[index]}`;
};
const formatDate = (value) => value ? new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "short" }).format(new Date(value)) : "—";
const metricPercent = (value) => typeof value === "number" ? percent(value) : "—";
const formatNumber = (value) => typeof value === "number" ? value.toLocaleString() : "—";
const stageLabel = (stage) => ({ baseline: "仅基准训练", analyze: "边界分析", full: "完整修复" }[stage] || stage);
const VIEW_SECTIONS = {
  overview: ["overview", "boundaries", "boundary-detail", "job-console", "roadmap"],
  new: ["experiment-config"],
  runs: ["runs", "run-detail"],
  catalog: ["catalog", "projects"],
  comparison: ["comparison"],
  reports: ["report-center"],
  system: ["system-check", "job-console"],
};
const VIEW_TITLES = { overview: "项目总览", new: "新建实验", runs: "实验记录", catalog: "模型与数据集", comparison: "对比分析", reports: "报告中心", system: "系统检查" };

function currentView() {
  if (window.location.pathname.startsWith("/runs/")) return "runs";
  const match = window.location.pathname.match(/^\/dashboard\/([^/]+)$/);
  return ({ overview: "overview", new: "new", runs: "runs", catalog: "catalog", comparison: "comparison", reports: "reports", system: "system" })[match?.[1]] || "overview";
}

function applyView(view = currentView()) {
  const managed = new Set(Object.values(VIEW_SECTIONS).flat());
  managed.forEach((id) => {
    const node = document.getElementById(id);
    if (node) node.classList.toggle("view-hidden", !VIEW_SECTIONS[view].includes(id));
  });
  document.querySelectorAll(".nav-item[data-view]").forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
  });
  document.title = `FailureAtlas · ${VIEW_TITLES[view]}`;
}

async function fetchJSON(url, options) {
  if (STATIC_SHOWCASE && url === "/api/dashboard") url = "data/dashboard.json";
  if (STATIC_SHOWCASE && url === "/api/jobs/current") {
    return { status: "idle", action: null, logs: [] };
  }
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `请求失败（${response.status}）`);
  }
  return response.json();
}

function experiment(name) {
  return state.data.experiments.find((item) => item.name === name);
}

function renderSummary() {
  const data = state.data;
  const delta = data.clean_accuracy_after - data.clean_accuracy_before;
  $("#model-meta").textContent = `${data.model} · ${data.dataset} · ${data.sample_count.toLocaleString()} 张测试图像`;
  $("#clean-after").textContent = percent(data.clean_accuracy_after);
  $("#clean-delta").textContent = `${signedPercent(delta)} 相对基线`;
  $("#macro-f1").textContent = metricPercent(data.clean_macro_f1_after);
  $("#clean-ece").textContent = typeof data.clean_ece_after === "number" ? data.clean_ece_after.toFixed(4) : "—";
  $("#parameter-count").textContent = formatNumber(data.parameter_count);
  $("#inference-speed").textContent = typeof data.inference_ms_per_sample === "number" ? `${data.inference_ms_per_sample.toFixed(3)} ms` : "未采集";
  $("#guardrail-value").textContent = signedPercent(delta);
  if (data.deployment_mode === "showcase") {
    $("#showcase-notice").hidden = false;
    $("#repair-button").disabled = true;
    $("#repair-button").textContent = "在线只读演示";
    $("#import-button").disabled = true;
    $("#rescan-button").disabled = true;
    $("#workspace-import-button").disabled = true;
  }
}

function renderTabs() {
  const host = $("#perturbation-tabs");
  host.innerHTML = state.data.experiments.map((item) => `
    <button class="tab" role="tab" aria-selected="${item.name === state.selected}" data-name="${item.name}">${LABELS[item.name]}</button>
  `).join("");
  host.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-name]");
    if (!button) return;
    state.selected = button.dataset.name;
    host.querySelectorAll(".tab").forEach((tab) => tab.setAttribute("aria-selected", tab === button));
    renderChart();
  });
}

function svgElement(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function renderChart() {
  const item = experiment(state.selected);
  const svg = $("#comparison-chart");
  const width = Math.max(svg.clientWidth, 520);
  const height = 330;
  const margin = { top: 20, right: 18, bottom: 42, left: 48 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = item.baseline.map((entry) => entry.severity);
  const minX = Math.min(...values), maxX = Math.max(...values);
  const allY = [...item.baseline, ...item.repaired].map((entry) => entry.accuracy);
  const minY = Math.max(0, Math.floor((Math.min(...allY) - .05) * 10) / 10);
  const maxY = 1;
  const x = (value) => margin.left + ((value - minX) / (maxX - minX || 1)) * innerWidth;
  const y = (value) => margin.top + (1 - (value - minY) / (maxY - minY)) * innerHeight;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";

  for (let step = 0; step <= 4; step++) {
    const value = minY + ((maxY - minY) * step) / 4;
    const yPos = y(value);
    svg.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: yPos, y2: yPos, class: "grid" }));
    const label = svgElement("text", { x: margin.left - 10, y: yPos + 4, "text-anchor": "end" });
    label.textContent = `${Math.round(value * 100)}%`;
    svg.append(label);
  }
  // Calibrated grids are dense near the clean setting, so only label points that
  // are far enough apart to stay readable.
  let lastLabelX = -Infinity;
  values.forEach((value) => {
    const position = x(value);
    if (position - lastLabelX < 52) return;
    lastLabelX = position;
    const label = svgElement("text", { x: position, y: height - 14, "text-anchor": "middle" });
    label.textContent = `${Number(value.toPrecision(3))}${UNIT[state.selected]}`;
    svg.append(label);
  });

  for (const [series, className, pointClass] of [[item.baseline, "baseline-line", "point-baseline"], [item.repaired, "repaired-line", "point-repaired"]]) {
    const path = series.map((entry, index) => `${index ? "L" : "M"}${x(entry.severity)},${y(entry.accuracy)}`).join(" ");
    svg.append(svgElement("path", { d: path, class: className }));
    series.forEach((entry) => svg.append(svgElement("circle", { cx: x(entry.severity), cy: y(entry.accuracy), r: 4, class: pointClass })));
  }

  const lastBefore = item.baseline.at(-1).accuracy;
  const lastAfter = item.repaired.at(-1).accuracy;
  const severeBefore = item.boundaries_before.severe;
  const severeAfter = item.boundaries_after.severe;
  $("#chart-summary").innerHTML = `
    <div><span>严重边界</span><strong>${boundary(severeBefore, item.name)} → ${boundary(severeAfter, item.name)}</strong></div>
    <div><span>最大强度变化</span><strong>${signedPercent(lastAfter - lastBefore)}</strong></div>
    <div><span>修复后末端精度</span><strong>${percent(lastAfter)}</strong></div>
  `;
}

function renderAgent() {
  $("#action-list").innerHTML = state.data.actions.map((action) => `
    <li><div><strong>${LABELS[action.perturbation]}</strong><p>增强概率 ${(action.probability * 100).toFixed(0)}%，上限 <code>${action.max_severity}</code><br>脆弱度 ${action.vulnerability_score.toFixed(3)}</p></div></li>
  `).join("");
}

function renderTable() {
  $("#boundary-table").innerHTML = state.data.experiments.map((item) => {
    const before = item.boundaries_before.severe;
    const after = item.boundaries_after.severe;
    const delta = item.repaired.at(-1).accuracy - item.baseline.at(-1).accuracy;
    const improved = before != null && (after == null || after > before);
    const conclusion = improved ? ["边界后移", "positive"] : delta < -.005 ? ["存在权衡", "negative"] : ["基本持平", "neutral"];
    return `<tr><td><strong>${LABELS[item.name]}</strong></td><td>${boundary(before, item.name)}</td><td>${boundary(after, item.name)}</td><td>${signedPercent(delta)}</td><td><span class="outcome ${conclusion[1]}">${conclusion[0]}</span></td></tr>`;
  }).join("");
}

function renderProjects() {
  const host = $("#project-list");
  if (!state.projects.length) {
    host.innerHTML = `<div class="import-status">还没有模型项目。导入一个 .fatlas 模型包开始校验。</div>`;
    return;
  }
  host.innerHTML = state.projects.map((project) => `
    <article class="project-row">
      <div class="project-identity">
        <span class="project-symbol" aria-hidden="true">${project.built_in ? "FA" : "TS"}</span>
        <span><strong>${escapeHTML(project.name)}</strong><small>${project.built_in ? "内置完整分析" : "PyTorch Export 导入模型"}</small></span>
      </div>
      <div class="project-stat"><span>干净准确率</span><strong>${percent(project.clean_accuracy)}</strong></div>
      <div class="project-stat"><span>测试样本</span><strong>${project.sample_count.toLocaleString()}</strong></div>
      <div class="project-stat"><span>输入形状</span><strong>${project.input_shape.slice(1).join(" × ")}</strong></div>
      <span class="project-status ${project.status}">${project.status === "analyzed" ? "已完成分析" : "校验通过"}</span>
    </article>
  `).join("");
}

function compatibleDatasets(modelId) {
  if (!state.catalog) return [];
  const allowed = new Set(state.catalog.compatibility.filter((item) => item.model === modelId && item.compatible).map((item) => item.dataset));
  return state.catalog.datasets.filter((dataset) => allowed.has(dataset.dataset_id));
}

function updateSelectionStatus() {
  const modelId = $("#experiment-model").value;
  const datasetId = $("#experiment-dataset").value;
  const model = state.catalog.models.find((item) => item.id === modelId);
  const dataset = state.catalog.datasets.find((item) => item.dataset_id === datasetId);
  const status = $("#selection-status");
  if (!model || !dataset) {
    status.textContent = "请选择有效的模型与数据集。";
    return;
  }
  const cacheText = dataset.cached ? "数据已缓存" : "首次运行将下载数据";
  status.innerHTML = `<strong>${escapeHTML(model.name)} × ${escapeHTML(dataset.display_name)}</strong><span>${model.parameters.toLocaleString()} 参数 · ${dataset.train_samples.toLocaleString()} 训练样本 · ${cacheText}</span>`;
  schedulePreflight();
}

function experimentPayload() {
  const payload = {
    model: $("#experiment-model").value,
    dataset: $("#experiment-dataset").value,
    stage: $("#experiment-stage").value,
    seed: Number($("#experiment-seed").value || 42),
    severity_scale: Number($("#experiment-severity").value || 1),
    reuse_artifacts: $("#experiment-reuse").value === "true",
  };
  const runId = $("#experiment-run-id").value.trim();
  const epochs = $("#experiment-epochs").value;
  const batchSize = $("#experiment-batch-size").value;
  const perturbations = [...document.querySelectorAll("#experiment-perturbations input:checked")].map((input) => input.value);
  if (runId) payload.run_id = runId;
  if (epochs) payload.epochs = Number(epochs);
  if (batchSize) payload.batch_size = Number(batchSize);
  if (payload.stage !== "baseline") payload.perturbations = perturbations;
  return payload;
}

function renderPreflight(result) {
  const host = $("#preflight-panel");
  const stepText = result.steps.map((step) => `<li class="${step.will_run ? "run" : "reuse"}"><span>${escapeHTML(step.title)}</span><strong>${step.will_run ? "执行" : "复用"}</strong></li>`).join("");
  host.innerHTML = `<div><strong>运行预检通过</strong><span>预计约 ${result.estimated_minutes} 分钟 · ${result.dataset_cached ? "数据已缓存" : "需下载数据"} · ${result.weights_ready ? "已有基准权重" : "尚无基准权重"}</span></div><ol>${stepText}</ol>`;
  host.className = "preflight-panel ready";
}

async function loadPreflight() {
  if (!state.catalog || STATIC_SHOWCASE) return;
  const host = $("#preflight-panel");
  host.className = "preflight-panel loading";
  host.innerHTML = `<span>正在检查兼容性、缓存与可复用产物…</span>`;
  try {
    renderPreflight(await fetchJSON("/api/preflight", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(experimentPayload()) }));
  } catch (error) {
    host.className = "preflight-panel error";
    host.innerHTML = `<strong>预检未通过</strong><span>${escapeHTML(error.message)}</span>`;
  }
}

function schedulePreflight() {
  clearTimeout(state.preflightTimer);
  state.preflightTimer = setTimeout(loadPreflight, 180);
}

function updateExperimentMode() {
  const baselineOnly = $("#experiment-stage").value === "baseline";
  $("#experiment-severity").disabled = baselineOnly;
  document.querySelectorAll("#experiment-perturbations input").forEach((input) => { input.disabled = baselineOnly; });
  schedulePreflight();
}

function syncDatasetOptions() {
  const datasets = compatibleDatasets($("#experiment-model").value);
  $("#experiment-dataset").innerHTML = datasets.map((dataset) => `<option value="${dataset.dataset_id}">${escapeHTML(dataset.display_name)}${dataset.cached ? " · 已缓存" : " · 待下载"}</option>`).join("");
  updateSelectionStatus();
}

function renderExperimentForm() {
  const modelSelect = $("#experiment-model");
  modelSelect.innerHTML = state.catalog.models.map((model) => `<option value="${model.id}">${escapeHTML(model.name)} · ${model.architecture}</option>`).join("");
  $("#experiment-perturbations").innerHTML = state.catalog.perturbations.map((item) => `<label><input type="checkbox" value="${item.id}" checked><span>${escapeHTML(item.display_name)}</span></label>`).join("");
  syncDatasetOptions();
  updateExperimentMode();
  if (STATIC_SHOWCASE) {
    $("#experiment-submit").disabled = true;
    $("#experiment-submit").textContent = "在线演示只读";
    $("#preflight-panel").innerHTML = `<span>在线演示不执行本地预检；请在本地版配置并启动实验。</span>`;
  }
}

function renderCatalog() {
  const models = state.catalog.models.map((model) => `
    <article class="catalog-item">
      <div><strong>${escapeHTML(model.name)}</strong><span>${escapeHTML(model.architecture)} · ${model.weights_ready ? "权重就绪" : "尚无权重"}</span></div>
      <dl><div><dt>参数量</dt><dd>${model.parameters.toLocaleString()}</dd></div><div><dt>输入</dt><dd>${model.input_shape.join(" × ")}</dd></div><div><dt>兼容数据集</dt><dd>${model.datasets.map((id) => state.catalog.datasets.find((item) => item.dataset_id === id)?.display_name || id).join("、")}</dd></div><div><dt>实验</dt><dd>${model.experiment_count ?? 0}</dd></div></dl>
    </article>
  `).join("");
  const datasetCards = state.catalog.datasets.map((dataset) => `
    <article class="catalog-item dataset-item">
      <div><strong>${escapeHTML(dataset.display_name)}</strong><span>${dataset.cached ? "本地缓存就绪" : "首次实验时下载"}</span></div>
      <dl><div><dt>训练/测试</dt><dd>${formatNumber(dataset.train_samples)} / ${formatNumber(dataset.test_samples)}</dd></div><div><dt>输入与类别</dt><dd>${dataset.input_channels} × ${dataset.input_size.join(" × ")} · ${dataset.num_classes} 类</dd></div><div><dt>归一化</dt><dd>μ ${dataset.normalization_mean.join(", ")} · σ ${dataset.normalization_std.join(", ")}</dd></div><div><dt>扰动 / 实验</dt><dd>${dataset.perturbations.length} 类 / ${dataset.experiment_count ?? 0} 次</dd></div></dl>
    </article>
  `).join("");
  $("#model-catalog").innerHTML = models + datasetCards;
  const datasets = state.catalog.datasets;
  $("#compatibility-head").innerHTML = `<tr><th>模型</th>${datasets.map((dataset) => `<th>${escapeHTML(dataset.display_name)}<small class="cache-dot ${dataset.cached ? "ready" : "pending"}">${dataset.cached ? "已缓存" : "待下载"}</small></th>`).join("")}</tr>`;
  $("#compatibility-body").innerHTML = state.catalog.models.map((model) => `<tr><td><strong>${escapeHTML(model.name)}</strong></td>${datasets.map((dataset) => {
    const compatible = state.catalog.compatibility.some((item) => item.model === model.id && item.dataset === dataset.dataset_id && item.compatible);
    return `<td><span class="compatibility ${compatible ? "yes" : "no"}">${compatible ? "兼容" : "—"}</span></td>`;
  }).join("")}</tr>`).join("");
}

function renderRuns() {
  const query = $("#run-search").value.trim().toLowerCase();
  const model = $("#run-model-filter").value;
  const dataset = $("#run-dataset-filter").value;
  const status = $("#run-status-filter").value;
  const date = $("#run-date-filter").value;
  const localDate = (value) => {
    if (!value) return "";
    const parsed = new Date(value);
    const pad = (part) => String(part).padStart(2, "0");
    return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())}`;
  };
  const visible = state.runs.filter((run) =>
    (!query || run.run_id.toLowerCase().includes(query)) &&
    (!model || run.model === model) &&
    (!dataset || run.dataset === dataset) &&
    (!status || run.status === status) &&
    (!date || localDate(run.created_at) === date)
  );
  $("#run-count").textContent = `${visible.length} / ${state.runs.length} 条记录`;
  const host = $("#runs-table");
  if (!visible.length) {
    host.innerHTML = `<tr><td colspan="9" class="empty-cell">没有符合筛选条件的实验。调整筛选器，或启动一个新实验。</td></tr>`;
    updateCompareButton();
    return;
  }
  host.innerHTML = visible.map((run) => `<tr>
    <td class="select-column"><input class="run-select" type="checkbox" aria-label="选择 ${escapeHTML(run.run_id)}" data-run-id="${escapeHTML(run.run_id)}" ${state.selectedRuns.has(run.run_id) ? "checked" : ""} ${run.status !== "completed" ? "disabled" : ""}></td>
    <td><button class="run-link" type="button" data-detail-id="${escapeHTML(run.run_id)}"><code>${escapeHTML(run.run_id)}</code></button></td>
    <td>${escapeHTML(run.model)}</td><td>${escapeHTML(run.dataset)}</td><td>${escapeHTML(stageLabel(run.stage))}</td>
    <td>${metricPercent(run.metrics?.accuracy)}</td>
    <td><span class="outcome ${run.status === "completed" ? "positive" : run.status === "failed" ? "negative" : "neutral"}">${run.status === "completed" ? "已完成" : run.status === "failed" ? "失败" : escapeHTML(run.status)}</span></td>
    <td>${formatDate(run.created_at)}</td><td><button class="text-action compact" type="button" data-detail-id="${escapeHTML(run.run_id)}">查看</button></td>
  </tr>`).join("");
  updateCompareButton();
}

function populateRunFilters() {
  const options = (values) => [...new Set(values)].sort().map((value) => `<option value="${escapeHTML(value)}">${escapeHTML(value)}</option>`).join("");
  $("#run-model-filter").innerHTML = `<option value="">全部模型</option>${options(state.runs.map((run) => run.model))}`;
  $("#run-dataset-filter").innerHTML = `<option value="">全部数据集</option>${options(state.runs.map((run) => run.dataset))}`;
}

function updateCompareButton() {
  const count = state.selectedRuns.size;
  const button = $("#compare-selected");
  button.disabled = count < 2 || count > 6;
  button.textContent = count ? `对比所选实验（${count}）` : "对比所选实验";
}

function renderReports() {
  const reports = state.runs.filter((run) => run.report_available);
  $("#report-list").innerHTML = reports.length ? reports.map((run) => `
    <article class="report-row">
      <div><strong>${escapeHTML(run.run_id)}</strong><span>${escapeHTML(run.model)} × ${escapeHTML(run.dataset)} · ${formatDate(run.completed_at || run.created_at)}</span></div>
      <div><span>${run.artifact_count} 个产物</span><a class="button secondary compact-button" href="/api/runs/${encodeURIComponent(run.run_id)}/artifacts/agent_report.md" target="_blank">查看报告</a><button class="text-action compact" data-detail-id="${escapeHTML(run.run_id)}" type="button">全部产物</button></div>
    </article>
  `).join("") : `<div class="comparison-empty">尚无可用报告。完整实验结束后会自动出现在这里。</div>`;
}

async function openRunDetail(runId, pushRoute = true) {
  applyView("runs");
  const section = $("#run-detail");
  section.hidden = false;
  $("#run-detail-title").textContent = runId;
  $("#run-detail-meta").textContent = "正在读取实验产物…";
  $("#run-detail-metrics").innerHTML = `<div class="detail-loading">载入中</div>`;
  section.scrollIntoView({ behavior: "smooth", block: "start" });
  try {
    const run = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}`);
    if (pushRoute && window.location.pathname !== `/runs/${encodeURIComponent(runId)}`) history.pushState({ runId }, "", `/runs/${encodeURIComponent(runId)}`);
    $("#run-detail-meta").textContent = `${run.model} × ${run.dataset} · ${stageLabel(run.stage)} · ${formatDate(run.created_at)}`;
    const metrics = [
      ["Accuracy", metricPercent(run.metrics.accuracy)], ["Macro-F1", metricPercent(run.metrics.macro_f1)],
      ["ECE", typeof run.metrics.ece === "number" ? run.metrics.ece.toFixed(4) : "—"], ["参数量", formatNumber(run.metrics.parameters)],
      ["修复后 Accuracy", metricPercent(run.metrics.repaired_accuracy)], ["干净集变化", typeof run.metrics.clean_delta === "number" ? signedPercent(run.metrics.clean_delta) : "—"],
    ];
    $("#run-detail-metrics").innerHTML = metrics.map(([label, value]) => `<div><span>${label}</span><strong>${value}</strong></div>`).join("");
    $("#run-detail-boundaries").innerHTML = run.experiments.length ? run.experiments.map((item) => {
      const before = item.boundaries_before?.severe;
      const after = item.boundaries_after?.severe;
      const delta = typeof item.endpoint_before === "number" && typeof item.endpoint_after === "number" ? signedPercent(item.endpoint_after - item.endpoint_before) : "—";
      return `<tr><td>${escapeHTML(LABELS[item.name] || item.name)}</td><td>${boundary(before, item.name)}</td><td>${boundary(after, item.name)}</td><td>${delta}</td></tr>`;
    }).join("") : `<tr><td colspan="4" class="empty-cell">该实验尚未生成失效边界分析。</td></tr>`;
    $("#run-detail-artifacts").innerHTML = run.artifacts.length ? run.artifacts.map((artifact) => `<a href="${artifact.url.split("/").map((part, index) => index > 3 ? encodeURIComponent(part) : part).join("/")}" target="_blank"><span>${escapeHTML(artifact.name)}</span><small>${formatBytes(artifact.size)}</small></a>`).join("") : `<div class="comparison-empty">没有可下载的产物。</div>`;
    const metadata = { ...run.metadata, ...run.config };
    const metadataItems = [
      ["状态", run.status], ["阶段", stageLabel(run.stage)], ["随机种子", metadata.seed], ["批量大小", metadata.batch_size],
      ["强度倍率", metadata.severity_scale], ["Python", metadata.python], ["PyTorch", metadata.torch], ["平台", metadata.platform],
    ];
    $("#run-detail-metadata").innerHTML = metadataItems.map(([label, value]) => `<div><dt>${label}</dt><dd>${value == null ? "—" : escapeHTML(String(value))}</dd></div>`).join("");
    $("#run-detail-repairs").innerHTML = run.repair_actions.length ? run.repair_actions.map((action) => `<div><strong>${escapeHTML(LABELS[action.perturbation] || action.perturbation)}</strong><span>概率 ${action.probability ?? "—"} · 最大强度 ${action.max_severity ?? "—"}</span></div>`).join("") : `<div class="comparison-empty">该运行未执行修复阶段。</div>`;
    const plots = run.artifacts.filter((artifact) => artifact.name.toLowerCase().endsWith(".png"));
    $("#run-detail-plots").innerHTML = plots.map((plot) => `<figure><img src="${plot.url}" alt="${escapeHTML(runId)} 的 ${escapeHTML(plot.name)}"><figcaption>${escapeHTML(plot.name)}</figcaption></figure>`).join("");
    const error = $("#run-detail-error");
    error.hidden = !run.error;
    if (run.error) error.innerHTML = `<strong>实验未完成</strong><span>${escapeHTML(run.error)}</span><small>建议检查上方任务日志、数据缓存与 GPU 状态后，使用新的 Run ID 重试。</small>`;
  } catch (error) {
    $("#run-detail-meta").textContent = `无法读取：${error.message}`;
    $("#run-detail-metrics").innerHTML = "";
  }
}

async function compareSelectedRuns() {
  const runIds = [...state.selectedRuns];
  $("#comparison-state").textContent = "正在计算";
  try {
    const result = await fetchJSON("/api/comparison", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run_ids: runIds }) });
    $("#comparison-empty").hidden = true;
    $("#comparison-result").hidden = false;
    $("#comparison-state").textContent = `${result.runs.length} 个实验 · ${result.dataset}`;
    if (!result.comparable) {
      $("#comparison-empty").hidden = false;
      $("#comparison-empty").textContent = result.limitation;
    }
    $("#comparison-head").innerHTML = `<tr><th>指标</th>${result.runs.map((run) => `<th><code>${escapeHTML(run.run_id)}</code><small>${escapeHTML(run.model)}</small></th>`).join("")}</tr>`;
    const metricRows = [
      ["Accuracy", (run) => metricPercent(run.metrics.accuracy)], ["Macro-F1", (run) => metricPercent(run.metrics.macro_f1)],
      ["ECE", (run) => typeof run.metrics.ece === "number" ? run.metrics.ece.toFixed(4) : "—"], ["参数量", (run) => formatNumber(run.metrics.parameters)],
      ["修复后 Accuracy", (run) => metricPercent(run.metrics.repaired_accuracy)], ["干净集变化", (run) => typeof run.metrics.clean_delta === "number" ? signedPercent(run.metrics.clean_delta) : "—"],
    ];
    const perturbations = [...new Set(result.runs.flatMap((run) => run.experiments.map((item) => item.name)))];
    const boundaryRows = perturbations.map((name) => [`${LABELS[name] || name}严重边界`, (run) => {
      const item = run.experiments.find((entry) => entry.name === name);
      return item ? boundary(item.boundaries_after?.severe ?? item.boundaries_before?.severe, name) : "—";
    }]);
    $("#comparison-body").innerHTML = [...metricRows, ...boundaryRows].map(([label, formatter]) => `<tr><th>${escapeHTML(label)}</th>${result.runs.map((run) => `<td>${formatter(run)}</td>`).join("")}</tr>`).join("");
    $("#comparison").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    $("#comparison-state").textContent = "无法比较";
    $("#comparison-empty").hidden = false;
    $("#comparison-empty").textContent = error.message;
    $("#comparison-result").hidden = true;
    $("#comparison").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderSystem() {
  const system = state.system;
  const items = [
    ["Python", system.python || "—"],
    ["PyTorch", system.torch || "未安装"],
    ["计算设备", system.cuda ? system.gpu : "CPU"],
    ["磁盘可用", formatBytes(system.disk_free_bytes)],
    ["运行模式", system.mode === "local" ? "本地可写" : "在线只读"],
  ];
  $("#system-grid").innerHTML = items.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHTML(String(value))}</dd></div>`).join("");
}

async function loadWorkspaceData() {
  if (STATIC_SHOWCASE) {
    state.catalog = {
      models: [{ id: "fashion_cnn", name: "FashionCNN", architecture: "compact_cnn", input_shape: [1, 28, 28], parameters: 421834, datasets: ["fashion_mnist"], weights_ready: true, experiment_count: 1 }],
      datasets: [{ dataset_id: "fashion_mnist", display_name: "Fashion-MNIST", train_samples: 60000, test_samples: 10000, input_channels: 1, input_size: [28, 28], num_classes: 10, normalization_mean: [0.286], normalization_std: [0.353], perturbations: Object.keys(LABELS), cached: true, experiment_count: 1 }],
      perturbations: Object.entries(LABELS).map(([id, display_name]) => ({ id, display_name })),
      compatibility: [{ model: "fashion_cnn", dataset: "fashion_mnist", compatible: true }],
    };
    state.runs = [];
    state.system = { python: "—", torch: "—", cuda: false, disk_free_bytes: null, mode: "showcase" };
  } else {
    const [catalog, runs, system] = await Promise.all([fetchJSON("/api/catalog"), fetchJSON("/api/runs"), fetchJSON("/api/system")]);
    state.catalog = catalog;
    state.runs = runs.runs;
    state.system = system;
  }
  populateRunFilters();
  renderExperimentForm(); renderCatalog(); renderRuns(); renderReports(); renderSystem();
}

async function submitExperiment(event) {
  event.preventDefault();
  const button = $("#experiment-submit");
  const payload = experimentPayload();
  if (payload.stage !== "baseline" && !payload.perturbations.length) {
    $("#selection-status").innerHTML = `<strong>无法启动实验</strong><span>至少选择一种扰动。</span>`;
    return;
  }
  button.disabled = true;
  button.textContent = "正在提交…";
  try {
    await fetchJSON("/api/jobs/experiment", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    $("#selection-status").innerHTML = `<strong>实验已进入队列</strong><span>${escapeHTML(payload.model)} × ${escapeHTML(payload.dataset)} · 可在下方日志查看进度</span>`;
    pollJob();
  } catch (error) {
    $("#selection-status").innerHTML = `<strong>无法启动实验</strong><span>${escapeHTML(error.message)}</span>`;
    button.disabled = false;
  } finally {
    button.textContent = "启动实验";
  }
}

function escapeHTML(value) {
  const node = document.createElement("span");
  node.textContent = value;
  return node.innerHTML;
}

async function loadProjects() {
  try {
    if (STATIC_SHOWCASE) {
      state.projects = [{
        id: "fashion-mnist-built-in",
        name: "Fashion-MNIST repaired CNN",
        built_in: true,
        status: "analyzed",
        clean_accuracy: state.data.clean_accuracy_after,
        sample_count: state.data.sample_count,
        input_shape: [-1, 1, 28, 28],
      }];
      renderProjects();
      return;
    }
    const payload = await fetchJSON("/api/projects");
    state.projects = payload.projects;
    renderProjects();
  } catch (error) {
    const status = $("#import-status");
    status.hidden = false;
    status.className = "import-status error";
    status.textContent = `无法读取模型工作区：${error.message}`;
  }
}

async function importBundle(file) {
  const status = $("#import-status");
  status.hidden = false;
  status.className = "import-status";
  status.textContent = `正在校验 ${file.name}：检查文件结构、张量形状和模型输出…`;
  $("#import-button").disabled = true;
  $("#workspace-import-button").disabled = true;
  try {
    if (file.size > 100 * 1024 * 1024) throw new Error("模型包超过 100 MB 限制。");
    const result = await fetchJSON("/api/projects/import", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream", "X-Bundle-Name": file.name },
      body: file,
    });
    status.className = "import-status success";
    status.textContent = `${result.name} 校验通过：${result.sample_count.toLocaleString()} 个样本，干净准确率 ${percent(result.clean_accuracy)}。${result.duplicate ? "该模型包已存在。" : "已加入本地工作区。"}`;
    await loadProjects();
  } catch (error) {
    status.className = "import-status error";
    status.textContent = `导入失败：${error.message}`;
  } finally {
    $("#import-button").disabled = false;
    $("#workspace-import-button").disabled = false;
    $("#bundle-input").value = "";
  }
}

async function startJob(action) {
  const buttons = [$("#repair-button"), $("#rescan-button")];
  buttons.forEach((button) => button.disabled = true);
  try {
    await fetchJSON(`/api/jobs/${action}`, { method: "POST" });
    pollJob();
  } catch (error) {
    alert(error.message);
    buttons.forEach((button) => button.disabled = false);
  }
}

async function pollJob() {
  if (STATIC_SHOWCASE) return;
  clearTimeout(state.poller);
  try {
    const job = await fetchJSON("/api/jobs/current");
    const consoleNode = $("#job-console");
    consoleNode.className = `job-console ${job.status}`;
    const title = { idle: "实验队列空闲", running: "实验正在运行", success: "实验已完成", error: "实验运行失败" }[job.status];
    $("#job-title").textContent = title;
    const actionName = { repair: "自动修复", rescan: "边界重扫", experiment: "配置实验" }[job.action] || job.action;
    const elapsed = job.started_at ? Math.max(0, Math.round((new Date(job.finished_at || Date.now()) - new Date(job.started_at)) / 1000)) : null;
    $("#job-time").textContent = job.action ? `${actionName} · ${job.status}${elapsed != null ? ` · ${elapsed}s` : ""}` : "等待操作";
    const recovery = job.status === "error" ? "\n\n建议：检查最后一条错误、系统检查与数据缓存；修正后使用新的 Run ID 重试。" : "";
    $("#job-log").textContent = (job.logs.length ? job.logs.join("\n") : "从右上角启动自动修复，或重新扫描全部边界。") + recovery;
    $("#job-log").scrollTop = $("#job-log").scrollHeight;
    const running = job.status === "running";
    $("#repair-button").disabled = running;
    $("#rescan-button").disabled = running;
    if (running) state.poller = setTimeout(pollJob, 2000);
    if (job.status === "success") setTimeout(loadDashboard, 700);
  } catch (error) {
    $("#job-log").textContent = error.message;
  }
}

async function loadDashboard() {
  try {
    state.data = await fetchJSON("/api/dashboard");
    renderSummary(); renderTabs(); renderAgent(); renderTable(); renderChart();
    $("#loading-state").hidden = true;
    $("#error-state").hidden = true;
    $("#dashboard").hidden = false;
    await Promise.all([loadProjects(), loadWorkspaceData()]);
    pollJob();
    applyView();
    const route = window.location.pathname.match(/^\/runs\/([^/]+)$/);
    if (route) openRunDetail(decodeURIComponent(route[1]), false);
  } catch (error) {
    $("#loading-state").hidden = true;
    $("#dashboard").hidden = true;
    $("#error-state").hidden = false;
    $("#error-message").textContent = error.message;
  }
}

$("#repair-button").addEventListener("click", () => startJob("repair"));
$("#rescan-button").addEventListener("click", () => startJob("rescan"));
$("#import-button").addEventListener("click", () => $("#bundle-input").click());
$("#workspace-import-button").addEventListener("click", () => $("#bundle-input").click());
$("#bundle-input").addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) importBundle(file);
});
$("#experiment-model").addEventListener("change", syncDatasetOptions);
$("#experiment-dataset").addEventListener("change", updateSelectionStatus);
$("#experiment-stage").addEventListener("change", updateExperimentMode);
$("#experiment-perturbations").addEventListener("change", schedulePreflight);
["#experiment-epochs", "#experiment-seed", "#experiment-batch-size", "#experiment-severity", "#experiment-reuse", "#experiment-run-id"].forEach((selector) => {
  $(selector).addEventListener("change", schedulePreflight);
});
$("#experiment-form").addEventListener("submit", submitExperiment);
["#run-search", "#run-model-filter", "#run-dataset-filter", "#run-status-filter", "#run-date-filter"].forEach((selector) => {
  $(selector).addEventListener(selector === "#run-search" ? "input" : "change", renderRuns);
});
$("#runs-table").addEventListener("change", (event) => {
  const checkbox = event.target.closest(".run-select");
  if (!checkbox) return;
  if (checkbox.checked) state.selectedRuns.add(checkbox.dataset.runId);
  else state.selectedRuns.delete(checkbox.dataset.runId);
  updateCompareButton();
});
$("#runs-table").addEventListener("click", (event) => {
  const button = event.target.closest("[data-detail-id]");
  if (button) openRunDetail(button.dataset.detailId);
});
$("#report-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-detail-id]");
  if (button) openRunDetail(button.dataset.detailId);
});
$("#compare-selected").addEventListener("click", compareSelectedRuns);
$("#close-run-detail").addEventListener("click", () => { $("#run-detail").hidden = true; history.pushState({}, "", "/dashboard/runs"); applyView("runs"); });
window.addEventListener("popstate", () => {
  const route = window.location.pathname.match(/^\/runs\/([^/]+)$/);
  if (route) openRunDetail(decodeURIComponent(route[1]), false);
  else { $("#run-detail").hidden = true; applyView(); }
});
document.querySelectorAll(".nav-item[data-view]").forEach((link) => link.addEventListener("click", (event) => {
  event.preventDefault();
  history.pushState({}, "", link.href);
  $("#run-detail").hidden = true;
  applyView(link.dataset.view);
  window.scrollTo({ top: 0, behavior: "smooth" });
}));
window.addEventListener("resize", () => state.data && renderChart());
loadDashboard();
