# gpubench standard suite

A small, fixed set of GPU benchmarks for the Linux driver stack (Mesa hk, the
drm/asahi kernel driver and the GPU firmware interface), each with a Vulkan
executor (Linux) and a Metal executor (macOS) that do the same work with the
same dimensions and the same timing method.

It is not a conformance suite and not a shader compiler test. Shaders are
deliberately trivial and are checked to compile to near-identical G17P code on
both sides (see `FAIRNESS.md`). What differs between the two columns of a
comparison is the submission, completion, scheduling and memory-management
path: that is what the suite measures.

## Tests

| Test | What it does per submission | What it isolates |
| --- | --- | --- |
| `submit.empty` | one empty command buffer | submit ioctl, fence/timeline signal, completion IRQ path, CPU wake-up |
| `compute.light` | 1 workgroup x 32 threads, one store each | per-dispatch kernel/firmware overhead, round-trip latency |
| `compute.heavy` | `kind=alu`: 1024 x 256 threads, 8 FMA chains x 512 iterations (2.1 GFLOP); `kind=bw`: 64 MiB read + 64 MiB write (float4) | scheduling and GPU feed under load; throughput vs depth |
| `render.light` | 64x64 RGBA8 pass, load-op clear, `kind=tri` 1 triangle or `kind=clear` none | render-pass / tiler / firmware render path, render pairs |
| `render.heavy` | `kind=fill`: 16 blended full-screen layers at 720p/1080p/2160p; `kind=geom`: 262144 small opaque triangles at 1080p | fill and geometry throughput, render scheduling at depth |
| `render.tvb` | 1080p pass with N long thin triangles (8K/32K/128K) that cross many tiles; `dim=N` renders into an N x N target instead (Vulkan) | tiler memory (TVB) overflow, partial renders, heap growth; cold first-pass cost |
| `render.mesh` | one mesh-shader draw (`vkCmdDrawMeshTasksEXT` / `drawMeshThreadgroups`) into a res x res RGBA8 target, optionally with a task/object stage, per-primitive outputs or two views; see [render.mesh](#rendermesh) | mesh/task launch, mesh output (vertex vs primitive heavy), multiview mesh, the render path behind it |
| `sync.signal` | `kind=host`: CPU signals a semaphore a parked submission waits on; `kind=queue`: two queues ping-pong a semaphore (`sema=timeline` or `binary`) | wait-before-signal path, cross-queue dependency latency |
| `mem.buffer` | create+allocate+bind, map, touch every 16 KiB page, unmap, free; plus a full alloc -> GPU fill -> wait -> free cycle; 4 KiB .. 64 MiB | GEM/VM bind, mmap, page-fault and free path |
| `mem.image` | same for 2D RGBA8 images 256^2 .. 4096^2, cycle = alloc -> clear pass -> wait -> free | image allocation, layout and first-use path |
| `copy.bw` | 64 MiB `kind=buf2buf`, `buf2img` (4096x4096 RGBA8) or `fill` | copy/blit bandwidth |
| `diag.fill` | `vkCmdFillBuffer` / `fillBuffer` of `size_mib` (1..1024, default 64) MiB; `init=1` fills the buffer once before timing, `storage=private\|shared` picks device-local or host-visible memory | fill bandwidth against first-touch page cost and memory placement (diagnostic, not in the plan) |
| `concurrent.cr` | `mode=queues`: compute.heavy and render.heavy on two queues, alone and together; `mode=interleave`: compute.light and render.light alternating on one queue | compute/render overlap; render<->compute switch cost |
| `idle.wake` | one compute.light submission after `gap_ms` (2, 20, 200) of GPU idle, then one immediately after | GPU power-down / wake-up path in firmware and kernel (`wake_cost_us`) |
| `query` | compute.light with start/end timestamps (`kind=timestamp`) or render.light with an occlusion query (`kind=occlusion`), plus result readback | query/timestamp overhead and readback |

The optional [GL/GLES executor](gl/README.md) covers the render and compute subset with the same sizes, warm-up, windows, and statistics. It also adds matched Vulkan `render.heavy kind=opaque` and `kind=texture` cases. See its documentation for supported modes and API comparison limits.

Present/swapchain is not covered yet (headless first).

## Depth

"Depth" is submission concurrency: how many submissions are in flight at
once. It is always the *total* number in flight, so every mode is comparable
at the same N.

- `mode=queue` (the default axis, N = 1, 2, 4, 8, 16, 32, 64): one queue, a
  ring of N slots. Vulkan: one `VkQueue`, N command pools/buffers, one
  timeline semaphore; submission k signals value k, and before submitting k
  the CPU waits for value k-N. Metal: one `MTLCommandQueue`; before committing
  command buffer k the CPU calls `waitUntilCompleted` on command buffer k-N.
- `mode=queues` (N = 2, 4, 8): N queues in one process, one thread per queue,
  each with 1 submission in flight. Vulkan: N `VkQueue`s of the family (hk
  gives each its own kernel queue and firmware context); Metal: N
  `MTLCommandQueue`s. Skipped when the device has fewer queues than N.
- `mode=procs` (N = 2, 4, 8, 16): N processes, each with one queue and 1
  submission in flight. Each process has its own DRM file, VM and firmware
  contexts. Children are started one at a time (each finishes its device setup
  and warm-up before the next starts), then all start the measured phase at
  the same instant.

Submissions are independent: no barriers or semaphores between them (Vulkan),
and every resource is created with untracked hazard tracking (Metal), so
neither driver adds dependencies the other does not.

Each submission is recorded fresh on both sides (Vulkan: reset the slot's
command pool, record, submit; Metal: new command buffer with unretained
references, encode, commit), because Metal has no reusable command buffers.

## Timing method (both executors)

- **Warm-up:** `--warmup-ms` (default 300) of the same stream, discarded.
- **Runs:** `--runs` (default 5) consecutive windows of `--run-ms` (default
  250) each, measured on a continuous stream (no drain between windows).
  Completions are attributed to the window in which the CPU observes them.
- **Wall-clock metrics** (untimed pass: no timestamps at all):
  - `us_per_submit`: window length / completions in the window. The primary
    metric of every submission test. Its inverse is the submission rate.
  - `latency_us`: per submission, from just before `vkQueueSubmit`/`commit`
    to the moment the CPU wait for it returns. At depth 1 this is the round
    trip. At depth N it includes queueing (about N x `us_per_submit`).
  - `submit_cpu_us`: time inside `vkQueueSubmit` / `commit`.
  - `record_cpu_us`: time to record/encode the command buffer.
- **GPU metrics** (separate timed pass, `--timestamps on`, default on; 3
  windows): Vulkan writes a timestamp at the start and the end of each command
  buffer (host query reset, no in-stream reset commands); Metal reads
  `GPUStartTime`/`GPUEndTime` of each command buffer.
  - `gpu_us`: end - start per submission, where start is the later of the
    submission's own start stamp and the previous submission's end. Vulkan
    lets a driver write the top-of-pipe stamp as soon as the queue reaches it,
    so at depth >= 2 the raw start can precede earlier work finishing.
  - `gpu_gap_us`: start of submission k - end of k-1 on the same queue (GPU
    idle between back-to-back submissions; meaningful at depth >= 2). A
    negative value means the start stamp was written before k-1 finished.
  - `us_per_submit_timed`: wall-clock rate during the timed pass. On hk
    timestamps ride on command retirement, so the difference from
    `us_per_submit` is itself a measured cost.
- Percentiles: `median`, `p5` and `p95` are taken over the per-window values
  for rate metrics and over all individual samples for per-submission
  metrics; `n` is the number of values.
- All CPU times use the monotonic clock (`CLOCK_MONOTONIC` /
  `mach_absolute_time`).

## render.mesh

One draw per submission into a `res` x `res` RGBA8 colour target (a 2-layer
array for `views=2`), load-op clear to (0, 0, 0, 0), store, no depth, no
blending, no culling. Vulkan records `vkCmdDrawMeshTasksEXT` inside dynamic
rendering (with `viewMask` 0b11 for `views=2`); Metal encodes
`drawMeshThreadgroups:threadsPerObjectThreadgroup:threadsPerMeshThreadgroup:`
with an `MTLMeshRenderPipelineDescriptor` pipeline. Same streams, windows and
metrics as the other render tests, plus `mprim_s` (primitives rasterized per
second, all views: groups x P x views per draw). The Vulkan executor enables
`VK_EXT_mesh_shader` and core multiview for this test only and skips a row
whose feature or limit is missing (`reason` names it). The result's
`pipeline` object records the shaders, the workload counts, the device or
pipeline limits and (Vulkan, with `VK_KHR_pipeline_executable_properties`)
the compiler statistics of each stage.

Parameters (both executors, identical meaning; `suite/vk/src/gb_mesh.h`
holds the shared definition):

| Param | Values | Meaning |
| --- | --- | --- |
| `stage` | `mesh` (default), `task` | `task`: a task (Vulkan) / object (Metal) workgroup of 32 threads launches 32 mesh workgroups and passes the index of the first in a 4-byte payload; the mesh workgroup index is `payload.base + local workgroup id`. The total mesh workgroups equal `groups`, which must be a multiple of 32 |
| `wg` | `32` (default), `128` | threads per mesh workgroup |
| `shape` | `prim` (default), `vert` | topo=tri geometry, below |
| `perprim` | `0` (default), `1` | 1 adds per-primitive outputs, a flat `vec4` colour and a `uint` id (Vulkan `perprimitiveEXT`, Metal per-primitive data of the `mesh<>` type), which the fragment shader reads |
| `views` | `1` (default), `2` | 2: two views into the 2-layer target; the colour depends on the view |
| `topo` | `tri` (default), `line` | `line`: the coverage workload, below |
| `groups` | 1..65535, default 4096 | mesh workgroups per draw (per view) |
| `res` | 64..8192, default 1024 | target width and height |

Geometry (identical on both sides; positions are in pixels with the origin at
the top-left, converted to clip space as `p * 2/res - 1`, with y negated on
Metal so that both APIs hit the same pixels):

- Tiles (topo=tri): workgroup `g` draws into tile `g mod T` of a row-major
  grid of `tiles_x = res / (GW*4)` by `res / (GH*4)` tiles, `T` tiles in all,
  wrapping when `groups > T`. The tile holds a `GW` x `GH` grid of 4-pixel
  cells (`GW, GH` = 8, 4 for wg=32 and 16, 8 for wg=128).
- `shape=prim` (primitive heavy): V = wg vertices, P = 2*wg triangles.
  Thread t writes vertex t at cell corner `(t mod GW, t div GW)` and
  triangles 2t and 2t+1, the two halves of the quad whose top-left corner it
  owns (diagonal a-e of corners a, b / d, e); a thread on the last column or
  row splits the neighbouring quad along its other diagonal instead, so all
  triangles are small (8 px) and non-degenerate. Per-vertex data: position
  and one flat colour.
- `shape=vert` (vertex heavy): V = min(2*wg, 128) vertices, P = V/4
  triangles (wg=32: 64 and 16, wg=128: 128 and 32). Triangle p sits in cell
  2p of the tile; vertices 4p..4p+3 are that cell's four corners, the
  triangle uses the first three and the fourth is written but not referenced.
  Each vertex also carries four smooth `vec4` varyings (functions of its pixel
  position, index, group and colour) that the fragment shader adds into the
  colour. V is capped at 128 (not 256) because Metal limits one `metal::mesh`
  to 16 KB: 256 x 96 bytes would be 24 KB.
- `topo=line` (coverage): V = 2*wg, P = wg. Thread t of workgroup g writes
  vertices t and t + wg and line t, a 1-pixel horizontal segment over pixel
  `i = (g*wg + t) mod res^2`, `x = i mod res`, `y = i div res`, on the row
  centre `y + 0.5` from `x + 1/64` to `x + 1 + 1/64` (the last column runs
  from `x + 1 - 1/64` to `x - 1/64`, so nothing is clipped). The segment holds
  exactly one pixel centre (rectangle rule) and leaves exactly one diamond
  (diamond-exit rule). So `groups=32768 wg=32 res=1024` covers the 1024x1024
  target exactly once per view.
- Colour of workgroup g in view v: `h = lowbias32(2g + v)`, RGBA8 bytes
  `(h & 255, h >> 8 & 255, h >> 16 & 255, 255)`; lowbias32 is
  `x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; x ^= x >> 16`.
  With `perprim=1` the primitive colour is the same and the id is
  `g * P + primitive index`; the fragment shader outputs
  `id < 2^31 ? mix(colour, primitive colour, 0.5) : 0`, i.e. the colour.
- Push constants (Vulkan) / `setMeshBytes` (Metal): `tiles_x, tiles_total,
  res, res^2` as four `uint`s.

Estimated GPU time per draw on Neo (from `render.heavy kind=geom`, about
300 Mtri/s on macOS): the plan's primitive rows (262144 triangles) about
1 ms, the vertex rows (25 MB / 12.5 MB of vertex output) a few hundred us,
the coverage row (2M one-pixel lines) a few ms.

`--check` (topo=line only; other rows report `check_status:
"not_checkable"`): after the timed pass the executor reads back every layer
and compares each pixel with its expected colour, the colour of the last
workgroup whose span covers it, each channel within +-1, or the clear colour
where no span reaches. The row gets `check_status`, `check_pixels`,
`check_bad`, `check_missing_blocks` ((view, workgroup) spans still entirely at
the clear colour), `check_wrong_view` (bad pixels holding the other view's
colour) and `check_pass`; a failed check makes the row `"status": "fail"` and
the executor exit 1. In mode=procs each process checks its own target and the
row adds them up.

## Diagnostics

These flags are off by default and do not change any plan row when absent.
Rows they change carry `"diagnostic": [...]`; they are not perf rows.

- `--interval-log FILE`: one JSON line per completed command buffer, appended
  to FILE (to `FILE.<pid>` in each mode=procs child; the parent writes
  nothing): `pid, test, params, mode, ring, slot, pass` (0 untimed, 1 timed),
  `measured, iter` (submission index), `clock, gpu_start, gpu_end, host_submit,
  host_done, host_clock` (seconds). Metal: `GPUStartTime`/`GPUEndTime`, and
  host times from `mach_absolute_time`, all on the mach host clock
  (`mach_host`), comparable across processes. Vulkan: a top-of-pipe /
  all-commands timestamp pair in every submission, converted with
  `timestampPeriod`, on the device clock (`vk_device`; comparable across
  processes on one device, not with the host), and host times on
  `CLOCK_MONOTONIC`. Each configuration first writes a header line
  `{"header":"pipeline", ..., "pipeline": {...}}` (render.mesh: the row's
  `pipeline` object, else null). On Vulkan the timestamps change the untimed
  pass, so such rows are marked `"diagnostic": ["interval-log"]`; on Metal the
  times are free and the flag alone does not mark the row.
- `--check-each` (render.mesh topo=line, mode queue or procs): every
  submission renders into its own ring slot's target and copies it to a
  host-visible buffer at the end of the same command buffer (Vulkan: layout
  barriers and `vkCmdCopyImageToBuffer`; Metal: a fence and a blit); the CPU
  checks it when the submission completes, warm-up included. Totals go into
  the row as `check_each_*` (`iters`, `bad_iters` and the `--check` fields);
  with `--interval-log` each line also carries `check_bad,
  check_missing_blocks, check_wrong_view, check_pass`.
- `--count N`: every pass submits exactly N measured command buffers (after
  `--warmup-ms`) in one window that runs from the first measured submission to
  the last completion, instead of `--runs` x `--run-ms` windows. In mode=procs
  each child submits N; the merged window ends with the last child.
  `--timestamps off` skips the timed pass (on Vulkan `--interval-log` already
  stamps every submission).

## API equivalents and reading notes

- **Semaphores:** Vulkan timeline and binary semaphores have no exact Metal
  counterpart. `sync.signal` uses `MTLSharedEvent` for host signals and
  `MTLEvent` for queue-to-queue signals, for both `sema` values (the result
  carries `metal_sync`).
- **Timestamps:**
  - The `query kind=timestamp` test uses `vkCmdWriteTimestamp2` on Vulkan and
    stage-boundary counter sampling (`MTLCounterSampleBuffer`) on Metal.
  - The timed pass of every other test uses command-buffer start/end
    timestamps on Vulkan, and `GPUStartTime`/`GPUEndTime` on Metal.
  - Stock Mesa 26.2 hk on G14 does not bracket a compute dispatch with
    start/end timestamps (`gpu_us` stays around 24 us whatever the work);
    the tip tree does. Check that `gpu_us` tracks the work before trusting it.
- **Occlusion:** a non-precise occlusion query on Vulkan; the boolean
  visibility result mode on Metal.
- **Fills:** both backends write repeated `0x3f` bytes. Vulkan uses the
  32-bit pattern `0x3f3f3f3f` to match Metal's byte-valued `fillBuffer`.
- **Copies and fills:** both are driver-internal operations (hk uses meta
  compute kernels), so `copy.bw` measures the driver's selected implementation,
  including internal shader codegen.
- **Empty submissions:** `submit.empty` measures what the API does with an
  empty command buffer. hk completes it in userspace without a kernel job
  (about 1 us round trip on the host). Use `compute.light` for the cheapest
  submission that reaches the kernel and firmware.
- **Latency at depth N:** `latency_us` at depth N includes about N - 1
  queued submissions ahead of each one. Compare round-trip latency at depth 1,
  and throughput (`us_per_submit`) at depth.

## Layout

- `shaders/`: GLSL (compiled to SPIR-V at build time) and the MSL with the
  same algorithm. One file pair per shader. render.mesh has one GLSL file per
  stage (`mesh.mesh`, `mesh.task`, `mesh.frag`, built in variants with `-D`)
  and one MSL file (`mesh.metal`), which the Metal executor compiles per
  configuration with the same macros, apart from the main library.
- `vk/`: Vulkan executor `gpubench-vk` (C11, CMake, Vulkan loader + headers,
  glslang). Static libraries `gb_common` (CLI, stats, JSON, timing, child
  processes) and `gb_vk` (instance/device/queue, buffers, images, pipelines,
  the submission-stream engine) are exported for other lanes.
- `metal/`: Metal executor `gpubench-mtl` (Objective-C, one binary), built on
  a Mac with `tools/build-macos.sh`.
- `plan.txt`: the standard run plan (see below).
- `FAIRNESS.md`: compiled-code comparison per shader.

## Executor interface (common to `suite/` and `rt/`)

Every executor is a command-line binary. The Vulkan one is named
`gpubench-<lane>-vk` and the Metal one `gpubench-<lane>-mtl`, except the
standard suite's, which are `gpubench-vk` and `gpubench-mtl`.

```
<exe> --info                       print one JSON object: device and driver identity
<exe> --list                       print the test names, one per line
<exe> --test NAME [options]        run one test, print one result line per configuration
```

Options every executor accepts (unknown `--param` keys are an error):

| Option | Meaning |
| --- | --- |
| `--depths 1,2,4` | depth list; one result per depth (default: the test's own) |
| `--mode queue\|queues\|procs` | concurrency mode (default `queue`) |
| `--param key=value` | test parameter, repeatable (e.g. `kind=alu`, `res=1080p`) |
| `--warmup-ms N` | warm-up per configuration |
| `--runs N` | measured windows |
| `--run-ms N` | window length |
| `--timestamps on\|off` | run the timed pass |
| `--device N` | device index (Vulkan) |
| `--check` | verify the render target after the timed pass (render.mesh topo=line) |
| `--check-each`, `--interval-log FILE`, `--count N` | diagnostics, see [Diagnostics](#diagnostics) |

Output: each configuration prints exactly one line on stdout that is a JSON
object with `"schema": "gpubench-result/1"` (see `records/SCHEMA.md`).
Anything else goes to stderr. A configuration the device cannot run prints a
result with `"status": "skip"` and a `"reason"`; a configuration that fails
prints `"status": "fail"` and an `"error"`. The exit status is non-zero only
when the executor could not run at all, or (exit 1) when a `--check` or
`--check-each` failed.

## Plans and the runner

`suite/plan.txt` (and `rt/plan.txt` when it exists) list one executor
invocation per line:

```
# chunk   lane    test            arguments
core-a    suite   compute.light   --depths 1,2,4,8,16,32,64
```

`lane` selects the executor (`suite` -> `gpubench-vk`/`gpubench-mtl`, `rt`
-> `gpubench-rt-vk`/`gpubench-rt-mtl`). `chunk` groups lines into runs of a
couple of minutes so that a locked GPU run on Neo stays short.
`tools/run-suite.sh` (used by `tools/run-linux.sh` and `tools/run-macos.sh`)
reads every plan, runs the selected chunks and writes one record directory.
It needs only a POSIX shell, so it runs on a stock macOS install.

## Building

Linux (Vulkan):

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      [-DGB_VULKAN_INCLUDE=/path/to/Vulkan-Headers/include]
ninja -C build            # -> build/bin/gpubench-vk (and rt executors)
```

macOS (Metal), on a Mac with the command-line tools:

```
tools/build-macos.sh      # -> bin/macos/gpubench-mtl (and rt executors)
```

See `records/README.md` for running the suite and adding a record.
