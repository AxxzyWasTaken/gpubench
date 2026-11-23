import { buildComparisons, canCompare, compatible, relative, workloadKey } from "./comparison.js";

const DEVICES = [
  { id: "neo", name: "A18 Pro", model: "MacBook Neo" },
  { id: "m5-pro", name: "M5 Pro", model: "MacBook Pro" },
  { id: "m4-mini", name: "M4", model: "Mac mini" },
];
const APIS = [
  { id: "vulkan", name: "Vulkan", platform: "Linux" },
  { id: "gl-native", name: "OpenGL", platform: "Linux · native" },
  { id: "gl-zink", name: "OpenGL / Zink", platform: "Linux" },
  { id: "metal", name: "Metal", platform: "macOS" },
];
const TEST_NAMES = {
  "submit.empty": "Empty submission",
  "compute.light": "Light compute",
  "compute.heavy": "Heavy compute",
  "render.light": "Light rendering",
  "render.heavy": "Heavy rendering",
  "copy.bw": "Copy bandwidth",
  "render.tvb": "Tile vertex buffer",
  "mem.image": "Image allocation",
  "mem.buffer": "Buffer allocation",
  query: "Query overhead",
  "rt.scene": "Ray-tracing scene",
  "compile.pso": "Pipeline compilation",
};
const METRIC_NAMES = {
  us_per_submit: "Submission time",
  latency_us: "Completion latency",
  submit_cpu_us: "CPU submit time",
  record_cpu_us: "CPU recording time",
  gpu_us: "GPU time",
  gpu_gap_us: "GPU idle time",
  us_per_submit_timed: "Time per submission (timed)",
  gbps: "Bandwidth",
  gflops: "Arithmetic throughput",
  gpix_s: "Pixel throughput",
  mtri_s: "Triangle throughput",
  fps: "Frame rate",
};
const LOCAL_FEED = "./data/benchmarks.json";
const LIVE_FEED = "https://raw.githubusercontent.com/aurora-silicon/gpubench/gpu-dashboard/dashboard/site/data/benchmarks.json";
const number = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 5 });
const axisNumber = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });
const dateFormat = new Intl.DateTimeFormat("en", { dateStyle: "medium", timeZone: "UTC" });
const byId = (id) => document.getElementById(id);
const theme = matchMedia("(prefers-color-scheme: dark)");
const state = {
  device: "neo", comparisonDevice: "all", runs: [], comparisons: [],
  hidden: new Set(), pinned: new Map(), plot: null,
};
const comparisonSearch = byId("comparison-search");
const comparisonApi = byId("comparison-api");
const comparisonResult = byId("comparison-result");
const comparisonSort = byId("comparison-sort");
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const workloadSelect = byId("workload");
const metricSelect = byId("metric");
const viewSelect = byId("view");
const canvas = byId("chart");

function node(tag, content, className) {
  const element = document.createElement(tag);
  if (content != null) element.textContent = content;
  if (className) element.className = className;
  return element;
}

function workloadName(row) {
  const items = Object.entries(row.params);
  const main = row.lane === "rt" ? row.params.preset : row.params.kind;
  const test = TEST_NAMES[row.test] ?? row.test;
  const details = [
    main,
    row.params.res,
    ...items.filter(([key]) => !["kind", "preset", "res"].includes(key)).map(([key, value]) => `${key}=${value}`),
    row.mode === "queue" ? null : row.mode,
  ].filter(Boolean);
  return details.length ? `${test} · ${details.join(", ")}` : test;
}

function groupName(row) {
  if (row.lane === "rt") return "Ray tracing";
  const prefix = row.test.split(".")[0];
  return { render: "Graphics", compute: "Compute", compile: "Compilation", mem: "Memory", copy: "Memory" }[prefix] ?? "Other workloads";
}

function metricName(name) {
  return METRIC_NAMES[name] ?? name.replaceAll("_", " ");
}

function unitName(unit) {
  return unit === "us" ? "µs" : unit;
}

function displayValue(value, unit) {
  return `${number.format(value)} ${unitName(unit)}`;
}

function deviceRuns(api) {
  return state.runs.filter((run) => run.device === state.device && run.api === api);
}

function usableRows(rows) {
  const counts = new Map();
  for (const row of rows) counts.set(row.depth, (counts.get(row.depth) ?? 0) + 1);
  // Multiple source rows with the same key are explicit in the table.
  return rows.filter((row) => row.status === "ok" && counts.get(row.depth) === 1);
}

function recordsFor(key) {
  return APIS.map((api) => {
    const candidates = deviceRuns(api.id).map((run) => {
      const rows = run.rows.filter((row) => workloadKey(row) === key);
      return { run, rows, usable: usableRows(rows) };
    }).filter((record) => record.rows.length);
    const pinned = candidates.find((record) => state.pinned.get(api.id) === record.run.id);
    const current = pinned ?? candidates.find((record) => record.usable.length &&
      (viewSelect.value !== "relative" || canCompare(record.run))) ??
      candidates.find((record) => record.usable.length) ?? candidates[0];
    return { ...api, candidates, ...current };
  });
}

function renderDevices() {
  const all = { id: "all", name: "All chips" };
  byId("devices").replaceChildren(...[all, ...DEVICES].map((device) => {
    const button = node("button", null, "device");
    button.type = "button";
    button.setAttribute("aria-pressed", String(device.id === state.comparisonDevice));
    button.append(node("strong", device.name));
    if (device.model) button.append(node("span", device.model));
    button.addEventListener("click", () => {
      if (state.comparisonDevice === device.id) return;
      state.comparisonDevice = device.id;
      if (device.id !== "all") state.device = device.id;
      state.hidden.clear();
      state.pinned.clear();
      renderDevices();
      renderComparison();
      renderWorkloads();
      byId("devices").querySelector('[aria-pressed="true"]').focus({ preventScroll: true });
    });
    return button;
  }));
}

function compareDelta(ratio) {
  if (ratio == null) return "unmatched";
  if (ratio > 1) return "linux";
  if (ratio < 1) return "metal";
  return "tied";
}

function makeComparisonRows(runs) {
  return buildComparisons(runs).map((pair) => {
    const { row, run } = pair.linux ?? pair.metal;
    const device = DEVICES.find((item) => item.id === run.device);
    const api = APIS.find((item) => item.id === pair.api);
    if (!device || !api) return null;
    const params = Object.entries(row.params).map(([key, value]) => `${key}=${value}`);
    const configuration = [`Depth ${row.depth}`, row.mode !== "queue" && row.mode, ...params].filter(Boolean).join(" · ");
    const title = TEST_NAMES[row.test] ?? row.test;
    const searchable = [row.test, workloadName(row), device.name, device.model, api.name, groupName(row), configuration]
      .join(" ").toLowerCase();
    return { ...pair, row, device, api, title, configuration, searchable, outcome: compareDelta(pair.ratio) };
  }).filter(Boolean);
}

function renderComparison() {
  const available = state.comparisons.filter((entry) =>
    state.comparisonDevice === "all" || entry.device.id === state.comparisonDevice);
  for (const option of comparisonApi.options) {
    option.disabled = Boolean(option.value && !available.some((entry) => entry.api.id === option.value));
  }
  if (comparisonApi.selectedOptions[0]?.disabled) comparisonApi.value = "";

  const query = comparisonSearch.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const filtered = available.filter((entry) =>
    (!comparisonApi.value || comparisonApi.value === entry.api.id) &&
    query.every((term) => entry.searchable.includes(term)));
  const counts = { linux: 0, metal: 0, tied: 0, unmatched: 0 };
  for (const entry of filtered) counts[entry.outcome]++;
  const matched = counts.linux + counts.metal + counts.tied;
  const resultStats = [
    ["Matched", matched, null, ""],
    ["Linux faster", counts.linux, matched && percent.format(counts.linux / matched), "linux-lead"],
    ["Metal faster", counts.metal, matched && percent.format(counts.metal / matched), "metal-lead"],
    ["Exact ties", counts.tied, matched && percent.format(counts.tied / matched), ""],
  ];
  byId("comparison-summary").replaceChildren(...resultStats.map(([label, total, share, className]) => {
    const item = node("div");
    const value = node("dd", null, className);
    value.append(node("span", number.format(total)));
    if (share) value.append(node("small", share));
    item.append(node("dt", label), value);
    return item;
  }));

  const result = comparisonResult.value;
  const visible = filtered.filter((row) => !result ||
    (result === "matched" ? row.outcome !== "unmatched" : row.outcome === result));
  const order = comparisonSort.value;
  visible.sort((a, b) => {
    if (order !== "workload") {
      if ((a.ratio == null) !== (b.ratio == null)) return a.ratio == null ? 1 : -1;
      if (a.ratio != null && a.ratio !== b.ratio) return order === "linux" ? b.ratio - a.ratio : a.ratio - b.ratio;
    }
    return a.title.localeCompare(b.title) || a.row.depth - b.row.depth ||
      a.configuration.localeCompare(b.configuration) || a.device.name.localeCompare(b.device.name) ||
      a.api.name.localeCompare(b.api.name);
  });

  const extent = Math.ceil(visible.reduce((max, row) =>
    row.ratio == null ? max : Math.max(max, Math.abs(Math.log2(row.ratio))), 1));
  byId("axis-min").textContent = `${axisNumber.format(2 ** -extent)}×`;
  byId("axis-max").textContent = `${axisNumber.format(2 ** extent)}×`;
  byId("comparison-rows").replaceChildren(...visible.map((entry) => renderComparisonRow(entry, extent)));
  const empty = byId("comparison-empty");
  empty.hidden = Boolean(visible.length);
  empty.textContent = result === "matched" && counts.unmatched
    ? "No Metal matches for these filters. Switch Results to Unmatched to see the available measurements."
    : "No results for these filters.";
  byId("comparison-status").textContent =
    `${number.format(visible.length)} shown · ${number.format(matched)} matched · ${number.format(counts.unmatched)} unmatched`;
  byId("comparison-scroll").scrollTop = 0;
}

function renderComparisonRow(entry, extent) {
  const { row, device, api, ratio, configuration, linux, metal, linuxMetric, metalMetric } = entry;
  const result = node("tr");
  const workload = node("td");
  const inspect = node("button", entry.title, "comparison-workload");
  inspect.type = "button";
  inspect.title = `${row.test} · ${configuration} · ${device.name} · ${api.name}`;
  inspect.setAttribute("aria-label", `Inspect ${inspect.title}`);
  inspect.addEventListener("click", () => inspectComparison(entry));
  const detail = node("span", configuration, "comparison-params technical");
  detail.title = configuration;
  workload.append(inspect, detail, node("span", `${device.name} · ${api.name}`, "comparison-mobile-stack"));

  const stack = node("td", null, "comparison-stack");
  stack.append(node("span", api.name), node("span", device.name, "comparison-chip"));
  const linuxValue = linuxMetric ? displayValue(linuxMetric.median, linuxMetric.unit) : "—";
  const metalValue = metalMetric ? displayValue(metalMetric.median, metalMetric.unit) : "—";
  const mobileValues = node("span", null, "comparison-mobile technical");
  mobileValues.append(node("span", `L ${linuxValue}`), node("span", `M ${metalValue}`));
  workload.append(mobileValues);
  const left = node("td", linuxValue, "comparison-median");
  const right = node("td", metalValue, "comparison-median");
  for (const [cell, source] of [[left, linux], [right, metal]]) {
    if (source) {
      cell.title = [metricName(source.row.primary ?? "unavailable"),
        dateFormat.format(new Date(source.run.date)), source.run.build, source.run.suite].filter(Boolean).join(" · ");
    }
  }

  const parity = node("td");
  if (ratio == null) {
    parity.append(node("div", entry.reason, "comparison-unmatched"));
  } else {
    const className = entry.outcome === "linux" ? "linux-lead" : entry.outcome === "metal" ? "metal-lead" : "";
    const content = node("div", null, `comparison-ratio ${className}`);
    const track = node("span", null, `parity-track ${className}`);
    track.setAttribute("aria-hidden", "true");
    const bar = node("span", null, "bar");
    const magnitude = Math.abs(Math.log2(ratio));
    bar.style.setProperty("--bar-size", `${Math.max(0.6, magnitude * 50 / extent)}%`);
    if (ratio === 1) bar.dataset.tie = "";
    track.append(bar);

    const precise = ratio === 1 ? "1.00" : ratio.toPrecision(Math.abs(ratio - 1) < 0.01 ? 5 : 3);
    const label = node("span", `${precise}×`, "comparison-number");
    const direction = { linux: "Linux faster", metal: "Metal faster", tied: "Exact tie" }[entry.outcome];
    label.setAttribute("aria-label", `${precise} times Metal, ${direction}`);
    content.title = `${precise}× Metal · ${direction}`;
    content.append(track, label);
    parity.append(content);
  }
  result.append(workload, stack, left, right, parity);
  return result;
}

function inspectComparison(entry) {
  state.device = entry.device.id;
  state.hidden.clear();
  state.pinned.clear();
  for (const source of [entry.linux, entry.metal]) {
    if (source) state.pinned.set(source.run.api, source.run.id);
  }
  viewSelect.value = entry.ratio == null ? "raw" : "relative";
  byId("workload-inspector").open = true;
  renderWorkloads(workloadKey(entry.row));
  if ([...metricSelect.options].some((option) => option.value === entry.row.primary)) {
    metricSelect.value = entry.row.primary;
    render();
  }
  byId("workload-inspector").scrollIntoView({ block: "start" });
  workloadSelect.focus({ preventScroll: true });
}

function renderWorkloads(previous = workloadSelect.value) {
  byId("inspector-device").textContent = DEVICES.find((device) => device.id === state.device).name;
  const unique = new Map();
  for (const run of state.runs.filter((record) => record.device === state.device)) {
    for (const row of run.rows) unique.set(workloadKey(row), row);
  }

  const groups = new Map();
  for (const [key, row] of unique) {
    const group = groupName(row);
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push([key, workloadName(row)]);
  }
  workloadSelect.replaceChildren(...[...groups].sort(([a], [b]) => a.localeCompare(b)).map(([group, entries]) => {
    const options = node("optgroup");
    options.label = group;
    options.append(...entries.sort((a, b) => a[1].localeCompare(b[1])).map(([key, title]) => new Option(title, key)));
    return options;
  }));
  const initial = workloadKey({ lane: "suite", test: "compute.heavy", params: { kind: "alu" }, mode: "queue" });
  workloadSelect.value = unique.has(previous) ? previous : unique.has(initial) ? initial : [...unique.keys()][0];
  render(true);
}

function render(resetMetrics = false) {
  const key = workloadSelect.value;
  const records = recordsFor(key);
  const row = records.find((record) => record.rows?.length)?.rows[0];
  if (!row) return;

  if (resetMetrics) {
    const previous = metricSelect.value;
    const names = new Map();
    let primary;
    for (const record of records) {
      for (const entry of record.usable ?? []) {
        primary ??= entry.primary;
        for (const name of Object.keys(entry.metrics)) names.set(name, metricName(name));
      }
    }
    metricSelect.replaceChildren(...[...names].map(([id, name]) => new Option(name, id)));
    metricSelect.disabled = !names.size;
    if (names.has(previous)) metricSelect.value = previous;
    else if (primary) metricSelect.value = primary;
  }
  const metric = metricSelect.value;
  const metal = records.find((record) => record.id === "metal");
  const metalValues = new Map((metal.usable ?? []).map((entry) => [entry.depth, entry.metrics[metric]]));
  const shape = records.flatMap((record) => record.usable ?? []).find((entry) => entry.metrics[metric])?.metrics[metric];
  const relativeView = viewSelect.value === "relative";
  const comparisonAvailable = canCompare(metal.run);
  const series = records.map((record) => {
    const points = (record.usable ?? []).flatMap((entry) => {
      const value = entry.metrics[metric];
      if (!value || !compatible(value, shape)) return [];
      const ratio = comparisonAvailable && canCompare(record.run) ? relative(value, metalValues.get(entry.depth)) : null;
      const y = relativeView ? (ratio == null ? null : Math.log2(ratio)) : value.median;
      return [{ depth: entry.depth, value, ratio, y }];
    });
    return { ...record, points };
  });

  if (relativeView) {
    const pairedDepths = new Set(series.filter((record) => record.id !== "metal")
      .flatMap((record) => record.points.filter((point) => point.ratio != null).map((point) => point.depth)));
    const baseline = series.find((record) => record.id === "metal");
    baseline.points = baseline.points.filter((point) => pairedDepths.has(point.depth));
  }
  const params = Object.entries(row.params).sort().map(([k, v]) => `${k}=${v}`);
  byId("workload-title").textContent = TEST_NAMES[row.test] ?? row.test;
  byId("workload-detail").textContent = [row.test, ...params, row.mode].join(" · ");
  byId("direction").textContent = relativeView ? "Metal = 1×" : shape ? `${unitName(shape.unit)} · ${shape.better} is better` : "";
  renderLegend(series);
  renderMeasurements(series);
  renderRecords(series);
  state.plot = { series, relativeView };
  byId("chart-tooltip").hidden = true;
  drawChart();
}

function renderLegend(series) {
  byId("legend").replaceChildren(...series.filter((record) => record.rows?.length).map((record) => {
    const button = node("button", null, "series");
    button.type = "button";
    const pressed = !state.hidden.has(record.id);
    button.setAttribute("aria-pressed", String(pressed));
    button.disabled = !record.points.length;
    button.title = record.platform;
    const swatch = node("span", null, `swatch ${record.id}`);
    swatch.setAttribute("aria-hidden", "true");
    button.append(swatch, node("span", record.name));
    button.addEventListener("click", () => {
      if (pressed) state.hidden.add(record.id);
      else state.hidden.delete(record.id);
      render();
      byId("legend").querySelectorAll("button").forEach((item) => {
        if (item.textContent === record.name) item.focus({ preventScroll: true });
      });
    });
    return button;
  }));
}

function renderMeasurements(series) {
  const rows = [];
  for (const record of series) {
    const counts = new Map();
    for (const row of record.rows ?? []) counts.set(row.depth, (counts.get(row.depth) ?? 0) + 1);
    for (const row of [...(record.rows ?? [])].sort((a, b) => a.depth - b.depth || a.line - b.line)) {
      const value = row.metrics?.[metricSelect.value];
      const point = record.points.find((item) => item.depth === row.depth);
      const status = { skip: "Skipped", fail: "Failed", ok: "Unavailable" }[row.status];
      const duplicate = counts.get(row.depth) > 1;
      const cells = [
        record.name,
        row.depth,
        value ? displayValue(value.median, value.unit) : status,
        value ? `${number.format(value.p5)}–${number.format(value.p95)} ${unitName(value.unit)}` : "—",
        value ? number.format(value.n) : "—",
        duplicate ? "Duplicate key" : point?.ratio == null || record.id === "metal" ? "—" : `${axisNumber.format(point.ratio)}×`,
      ];
      const tr = node("tr");
      tr.title = `Source row ${row.line}`;
      tr.replaceChildren(...cells.map((cell) => node("td", cell)));
      rows.push(tr);
    }
  }
  byId("measurements").replaceChildren(...rows);
  byId("measurement-count").textContent = rows.length ? `${rows.length} rows` : "";
}

function publicLink(source) {
  if (!source) return null;
  try {
    const url = new URL(source);
    return url.protocol === "https:" && !url.username && !url.password ? url.href : null;
  } catch {
    return null;
  }
}

function renderRecords(series) {
  const rows = [];
  for (const api of series) {
    const runs = deviceRuns(api.id);
    if (!runs.length && api.id === "gl-zink") continue;
    const display = api.run ?? runs[0];
    const entry = node("div", null, "run");
    entry.append(node("strong", `${api.name} · ${api.platform}`));
    if (!display) {
      entry.append(node("span", "No published run", "run-missing"));
    } else {
      const os = api.id === "metal" && display.osVersion ? `macOS ${display.osVersion}` : null;
      const meta = [os, display.build, display.mesa && `Mesa ${display.mesa.slice(0, 12)}`,
        display.suite ? `suite ${display.suite.slice(0, 16)}` : "suite unrecorded"].filter(Boolean);
      const description = node("span", meta.join(" · "), "technical run-meta");
      description.title = `Results SHA-256: ${display.sha256}`;
      entry.append(description, node("time", dateFormat.format(new Date(display.date)), "technical"));
      entry.lastChild.dateTime = display.date;
      const note = api.run ? display.note : ["No rows for this workload.", display.note].filter(Boolean).join(" ");
      if (note) entry.append(node("span", note, "run-note"));
      const source = publicLink(display.source);
      if (source) {
        const link = node("a", "Source record", "text-link run-link");
        link.href = source;
        entry.append(link);
      }
      if (api.candidates.length > 1) {
        const select = node("select", null, "run-picker technical");
        select.setAttribute("aria-label", `${api.name} source run`);
        select.append(new Option("Latest matching run", ""));
        for (const { run } of api.candidates) {
          select.append(new Option(`${dateFormat.format(new Date(run.date))} · ${run.build ?? "build unrecorded"}`, run.id));
        }
        select.value = state.pinned.get(api.id) ?? "";
        select.addEventListener("change", () => {
          if (select.value) state.pinned.set(api.id, select.value);
          else state.pinned.delete(api.id);
          render(true);
          byId("records").querySelectorAll("select").forEach((item) => {
            if (item.getAttribute("aria-label") === select.getAttribute("aria-label")) item.focus({ preventScroll: true });
          });
        });
        entry.append(select);
      }
    }
    rows.push(entry);
  }
  byId("records").replaceChildren(...rows);
}

function drawChart() {
  if (!state.plot) return;
  const { series, relativeView } = state.plot;
  const visible = series.filter((record) => !state.hidden.has(record.id));
  const plotted = visible.flatMap((record) => record.points.filter((point) => point.y != null));
  const empty = byId("chart-empty");
  canvas.hidden = !plotted.length;
  byId("chart-caption").hidden = canvas.hidden;
  empty.hidden = !canvas.hidden;
  if (!plotted.length) {
    empty.textContent = relativeView ? "No comparable Metal measurements for this selection." : "No measurements for the visible series.";
    return;
  }
  const depth = [...new Set(series.flatMap((record) => (record.rows ?? []).map((row) => row.depth)))].sort((a, b) => a - b);
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  if (!context) return;
  context.scale(ratio, ratio);
  const style = getComputedStyle(canvas);
  const color = (name) => style.getPropertyValue(`--${name}`).trim();
  const width = rect.width;
  const height = rect.height;
  const left = width > 500 ? 66 : 53;
  const right = width - 20;
  const top = 18;
  const bottom = height - 28;
  const values = plotted.map((point) => point.y);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const absolute = Math.max(Math.abs(min), max, 0.000001);
  const magnitude = 10 ** Math.floor(Math.log10(absolute / 4));
  const rawStep = [1, 2, 2.5, 5, 10].map((n) => n * magnitude).find((n) => n >= absolute / 4) ?? 10 * magnitude;
  const step = relativeView ? Math.max(.25, Math.ceil(absolute) / 2) : rawStep;
  const yMin = relativeView ? -2 * step : Math.floor(min / step) * step;
  const yMax = relativeView ? 2 * step : Math.max(step, Math.ceil(max / step) * step);
  const y = (value) => bottom - (value - yMin) / (yMax - yMin) * (bottom - top);
  const x = (index) => depth.length === 1 ? (left + right) / 2 : left + index / (depth.length - 1) * (right - left);
  state.plot.coordinates = { depth, x };
  const yTicks = Math.round((yMax - yMin) / step);
  context.font = `11px ${color("mono")}`;
  context.textBaseline = "middle";
  context.lineWidth = 1;
  for (let tick = 0; tick <= yTicks; tick++) {
    const value = yMin + tick * step;
    const pos = Math.round(y(value)) + .5;
    context.beginPath();
    context.setLineDash(relativeView && Math.abs(value) < 1e-9 ? [4, 3] : []);
    context.strokeStyle = relativeView && Math.abs(value) < 1e-9 ? color("muted") : color("grid");
    context.moveTo(left - 6, pos);
    context.lineTo(right + 5, pos);
    context.stroke();
    context.fillStyle = color("muted");
    context.textAlign = "right";
    context.fillText(relativeView ? `${axisNumber.format(2 ** value)}×` : axisNumber.format(Math.abs(value) < 1e-12 ? 0 : value), left - 10, pos);
  }
  context.setLineDash([]);
  context.textAlign = "center";
  depth.forEach((value, index) => {
    const pos = x(index);
    context.strokeStyle = color("grid");
    context.beginPath();
    context.moveTo(pos, bottom);
    context.lineTo(pos, bottom + 4);
    context.stroke();
    context.fillStyle = color("muted");
    context.fillText(value, pos, bottom + 16);
  });

  const highlighted = depth.indexOf(state.plot.hoverDepth);
  if (highlighted !== -1) {
    const pos = Math.round(x(highlighted)) + .5;
    context.beginPath();
    context.setLineDash([3, 4]);
    context.strokeStyle = color("muted");
    context.moveTo(pos, top);
    context.lineTo(pos, bottom);
    context.stroke();
  }

  // Draw Metal first so coincident Linux points remain easy to see.
  for (const record of [...visible].reverse()) {
    const points = new Map(record.points.map((point) => [point.depth, point.y]));
    context.strokeStyle = color(record.id);
    context.fillStyle = color(record.id);
    context.lineWidth = 1.75;
    context.setLineDash(record.id === "metal" ? [5, 4] : record.id === "gl-zink" ? [2, 3] : []);
    context.beginPath();
    let connected = false;
    depth.forEach((value, index) => {
      const point = points.get(value);
      if (point == null) {
        connected = false;
        return;
      }
      if (connected) context.lineTo(x(index), y(point));
      else context.moveTo(x(index), y(point));
      connected = true;
    });
    context.stroke();
    context.setLineDash([]);
    depth.forEach((value, index) => {
      const point = points.get(value);
      if (point == null) return;
      context.beginPath();
      context.arc(x(index), y(point), 3.2, 0, 2 * Math.PI);
      context.fill();
      context.strokeStyle = color("bg");
      context.lineWidth = 1;
      context.stroke();
    });
  }
}

function inspectPoint(event) {
  const plot = state.plot;
  if (!plot?.coordinates) return;
  const { depth, x } = plot.coordinates;
  const cursor = event.clientX - canvas.getBoundingClientRect().left;
  let closest = 0;
  for (let index = 1; index < depth.length; index++) {
    if (Math.abs(x(index) - cursor) < Math.abs(x(closest) - cursor)) closest = index;
  }
  if (plot.hoverDepth === depth[closest]) return;
  const tooltip = byId("chart-tooltip");
  const items = [];
  for (const record of plot.series) {
    if (state.hidden.has(record.id)) continue;
    const point = record.points.find((item) => item.depth === depth[closest] && item.y != null);
    if (!point) continue;
    const value = plot.relativeView ? `${axisNumber.format(point.ratio)}×` : displayValue(point.value.median, point.value.unit);
    const detail = node("div", null, "tooltip-series");
    detail.append(node("span", record.name), node("span", value, `technical tooltip-value ${record.id}`));
    items.push(detail);
  }
  tooltip.hidden = !items.length;
  plot.hoverDepth = items.length ? depth[closest] : null;
  if (items.length) {
    tooltip.replaceChildren(node("strong", `Depth ${depth[closest]}`), ...items);
    const left = x(closest);
    const width = tooltip.offsetWidth;
    const box = canvas.clientWidth;
    tooltip.style.setProperty("--tooltip-left", `${Math.max(0, left + width + 14 <= box ? left + 14 : left - width - 14)}px`);
  }
  drawChart();
}

async function fetchFeed(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Benchmark feed returned ${response.status}`);
  const feed = await response.json();
  if (feed.schema !== 1 || !Array.isArray(feed.runs) || !feed.runs.length) throw new Error("Invalid benchmark feed");
  for (const run of feed.runs) {
    if (!Array.isArray(run.rows) || !Number.isFinite(Date.parse(run.date))) throw new Error("Invalid run in benchmark feed");
  }
  return feed;
}

function applyFeed(feed) {
  state.runs = feed.runs.toSorted((a, b) => b.date.localeCompare(a.date));
  state.comparisons = makeComparisonRows(state.runs);
  const device = DEVICES.find((item) => state.runs.some((run) => run.device === item.id));
  if (!device) throw new Error("No supported devices in benchmark feed");
  if (!state.runs.some((run) => run.device === state.device)) state.device = device.id;
  byId("explorer").hidden = false;
  byId("load-status").hidden = true;
  renderDevices();
  renderComparison();
  renderWorkloads();
}

async function start() {
  const status = byId("load-status");
  status.hidden = false;
  status.textContent = "Loading benchmark runs…";
  const current = fetchFeed(LIVE_FEED, AbortSignal.timeout(6000)).catch(() => null);
  let snapshot;
  try {
    snapshot = await fetchFeed(LOCAL_FEED, AbortSignal.timeout(6000));
    applyFeed(snapshot);
  } catch {
    status.textContent = "Loading the live benchmark feed…";
  }
  const live = await current;
  try {
    if (live) {
      if (JSON.stringify(live) !== JSON.stringify(snapshot)) applyFeed(live);
      byId("download").href = LIVE_FEED;
    } else if (!snapshot) {
      throw new Error("Both benchmark feeds are unavailable");
    }
  } catch {
    if (!snapshot) status.textContent = "Benchmark runs are currently unavailable. Reload to retry.";
  }
}

comparisonSearch.addEventListener("input", renderComparison);
for (const filter of [comparisonApi, comparisonResult, comparisonSort]) {
  filter.addEventListener("change", renderComparison);
}
byId("comparison-title").nextElementSibling.addEventListener("click", () => { byId("methodology").open = true; });
canvas.addEventListener("pointermove", inspectPoint);
canvas.addEventListener("pointerdown", inspectPoint);
canvas.addEventListener("pointerleave", () => {
  if (!state.plot || state.plot.hoverDepth == null) return;
  state.plot.hoverDepth = null;
  byId("chart-tooltip").hidden = true;
  drawChart();
});
workloadSelect.addEventListener("change", () => { state.pinned.clear(); render(true); });
metricSelect.addEventListener("change", () => render());
viewSelect.addEventListener("change", () => render(true));
new ResizeObserver(drawChart).observe(canvas);
theme.addEventListener("change", drawChart);
document.fonts.ready.then(drawChart);
start();
