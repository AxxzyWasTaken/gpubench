import assert from "node:assert/strict";
import { test } from "node:test";
import { buildComparisons, relative } from "./site/comparison.js";

const rate = (median) => ({ median, unit: "fps", better: "higher" });
const time = (median) => ({ median, unit: "us", better: "lower" });

function result(median, extras = {}) {
  return {
    lane: "suite", test: "render.light", params: { kind: "draw", size: 32 },
    mode: "queue", depth: 1, status: "ok", primary: "fps",
    metrics: { fps: rate(median) }, ...extras,
  };
}

function run(api, rows, extras = {}) {
  return { api, device: "neo", date: "2026-11-21", suite: "123abcd", rows, ...extras };
}

test("ratios orient throughput and latency in the same direction", () => {
  assert.equal(relative(rate(160), rate(100)), 1.6);
  assert.equal(relative(time(50), time(100)), 2);
  assert.equal(relative(time(100), time(50)), 0.5);
  assert.equal(relative(time(50), rate(100)), null);
  assert.equal(relative(rate(0), rate(100)), null);
});

test("a case matches only within its device, workload, mode, parameters, depth and metric", () => {
  const metal = run("metal", [result(100, { params: { size: 32, kind: "draw" } })]);
  const linux = run("vulkan", [
    result(150),
    result(190, { depth: 2 }),
    result(200, { mode: "queues" }),
    result(120, { params: { kind: "clear", size: 32 } }),
    result(116, { lane: "other" }),
  ]);
  const otherDevice = run("vulkan", [result(210)], { device: "m4-mini" });
  const anotherMetric = run("gl-native", [result(70, { primary: "us", metrics: { us: time(70) } })]);
  const results = buildComparisons([metal, linux, otherDevice, anotherMetric]);
  assert.deepEqual(results.filter((entry) => entry.ratio != null).map((entry) => entry.ratio), [1.5]);
  assert.equal(results.length, 7);
  assert.equal(results.find((entry) => entry.api === "gl-native").reason, "Different primary metrics");
});

test("the newest recorded clean case wins and older runs never add duplicate pairs", () => {
  const metal = run("metal", [result(100)]);
  const old = run("vulkan", [result(140), result(120, { depth: 2 })], { date: "2026-11-19" });
  const next = run("vulkan", [result(160)], { date: "2026-11-22" });
  const unknown = run("vulkan", [result(300)], { date: "2026-11-23", suite: null });
  const dirty = run("vulkan", [result(240)], { date: "2026-11-24", suite: "567def0+dirty" });
  const results = buildComparisons([unknown, old, dirty, metal, next]);
  assert.deepEqual(results.map(({ ratio }) => ratio).sort(), [1.6, null]);
  assert.equal(results.find((entry) => entry.ratio != null).linux.run, next);
});

test("bad results in the newest clean run stay unmatched", () => {
  const metal = run("metal", [result(100), result(100, { depth: 2 })]);
  const old = run("vulkan", [result(150), result(150, { depth: 2 })], { date: "2026-11-19" });
  const next = run("vulkan", [result(170), result(180), result(0, { depth: 2, status: "skip" })], {
    date: "2026-11-22",
  });
  const results = buildComparisons([metal, old, next]);
  assert.deepEqual(results.map(({ ratio, reason }) => [ratio, reason]), [
    [null, "Duplicate result key"], [null, "Failed or skipped"],
  ]);
});
