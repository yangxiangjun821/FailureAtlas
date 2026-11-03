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

const state = { data: null, projects: [], selected: "rotation", poller: null };
const $ = (selector) => document.querySelector(selector);
const percent = (value, digits = 2) => `${(value * 100).toFixed(digits)}%`;
const signedPercent = (value) => `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
const boundary = (value, name) => value == null ? "未触发" : `${value}${UNIT[name]}`;

async function fetchJSON(url, options) {
  if (STATIC_SHOWCASE && url === "/api/dashboard") url = "/data/dashboard.json";
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
  $("#guardrail-value").textContent = signedPercent(delta);
  for (const [name, id] of [["rotation", "rotation-boundary"], ["gaussian_noise", "noise-boundary"], ["blur", "blur-boundary"]]) {
    const item = experiment(name);
    $("#" + id).textContent = `${boundary(item.boundaries_before.severe, name)} → ${boundary(item.boundaries_after.severe, name)}`;
  }
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
  values.forEach((value) => {
    const label = svgElement("text", { x: x(value), y: height - 14, "text-anchor": "middle" });
    label.textContent = `${value}${UNIT[state.selected]}`;
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
    $("#job-time").textContent = job.action ? `${job.action === "repair" ? "自动修复" : "边界重扫"} · ${job.status}` : "等待操作";
    $("#job-log").textContent = job.logs.length ? job.logs.join("\n") : "从右上角启动自动修复，或重新扫描全部边界。";
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
    await loadProjects();
    pollJob();
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
window.addEventListener("resize", () => state.data && renderChart());
loadDashboard();
