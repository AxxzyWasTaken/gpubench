# Workload equivalence and comparison limits

The Vulkan and Metal executors implement the same algorithms and dimensions. GLSL and MSL source are included under `shaders/`. Submission depths have the same total number of jobs in flight, and the standard submission tests use independent work without additional inter-submission hazards.

The standard suite reports an untimed pass and a separate GPU-timestamp pass. Instrumentation can change scheduling, query-result publication, and throughput. Compare `us_per_submit` with the same untimed metric on the other system; `us_per_submit_timed` measures the instrumented path. A timestamped GPU gap is not an uninstrumented utilization measurement.

API operations also differ in scope. Vulkan timeline/binary semaphores use Metal events as documented in [README.md](README.md). Empty Vulkan submissions can complete in userspace, so `compute.light` is the minimum useful kernel/firmware submission test. Public API timings do not directly measure kernel CPU time or display scanout.

For a driver comparison, verify matching shader work, compiler output, resource formats, concurrency, and operating conditions. The Vulkan executor exposes pipeline statistics with `--shader-stats`; the Metal executor can retain its own pipeline archives with `--dump-archive DIR`. Compiler differences, register use, and fixed-function setup can affect execution even when source algorithms match.

The current publication contains macOS measurement records. Linux performance comparisons will be published separately after qualification. See [record provenance](../records/README.md).

The optional GL/GLES executor shares render/compute shader algorithms and workload dimensions with Vulkan. GL fence/flush submission and implicit hazard handling have different API scopes from Vulkan command buffers and explicit synchronization. No compiled-code parity is claimed for GL or Zink. See [the GL timing notes](gl/README.md#timing-and-comparison-limits).

## render.mesh

What matches. Both executors take the workload from one definition
(`vk/src/gb_mesh.h`: parameters, counts, colour hash, the topo=line check) and
the GLSL (`shaders/mesh.mesh`, `mesh.task`, `mesh.frag`) and MSL
(`shaders/mesh.metal`) are built from the same macros (`WG`, `TOPO_LINE`,
`SHAPE_VERT`, `PERPRIM`, `VIEWS`, `TASK`), one compiled variant per
configuration on both sides. They declare the same output maxima
(`max_vertices`/`max_primitives` = the `mesh<V, P, NV, NP>` counts), the same
threads per workgroup, the same per-vertex varyings (a flat colour, plus four
smooth `vec4` for shape=vert) and per-primitive data (a flat `vec4` and a
`uint`), write the same positions, indices and colours from the same threads,
and read 16 bytes of constants (`tiles_x, tiles_total, res, res^2`). The task
(object) stage is 32 threads launching 32 mesh workgroups with a 4-byte
payload. The target is RGBA8, res x res (x 2 layers), load-op clear, store,
no depth, blending or culling. Submissions are independent (no barriers;
untracked hazard tracking on Metal) and each is recorded fresh, as in the
other render tests.

What does not match, or is not known to match:

- Multiview. Vulkan renders `views=2` with `viewMask` 0b11 and reads
  `gl_ViewIndex`; hk on G17P replays a multiview mesh draw once per view, so
  the mesh stage runs twice. Metal uses vertex amplification (count 2, view
  mappings with `renderTargetArrayIndexOffset` 0 and 1, `[[amplification_id]]`
  in the mesh function). Apple does not document whether a mesh threadgroup
  then runs once per view or once with per-view outputs, so the mesh-stage
  work per view may differ. Under amplification the Metal primitive also
  carries `[[render_target_array_index]]` = 0 (reported as needed for
  amplified mesh draws), 4 bytes per primitive that Vulkan does not write.
  `GB_MESH_MTL_LAYER_OUT=0` drops it; `GB_MESH_MTL_VIEWS=layer` instead draws
  twice the mesh groups with the layer in each primitive, the same work
  split as hk's replay. The result's `pipeline.workload.multiview`
  (`viewMask`, `amplification`, `layer`) and
  `pipeline.primitive_layer_output` say which ran.
- Layered target. The hk tree read for this note (4e03ea8d02c,
  `hk_g17p_compression.h`) compresses single-layer colour images on G17P but
  keeps arrays plain, so a `views=2` row may compare an uncompressed Vulkan
  target with a compressed Metal one. The Vulkan target always has
  `TRANSFER_SRC` usage (for `--check`), which that policy allows with
  compression; Metal blits need no usage flag.
- Indices. MSL mesh indices are `uchar` (fixed by the `mesh<>` type); GLSL
  writes `uvec3`/`uvec2` and the driver picks the storage.
- Output calls. GLSL calls `SetMeshOutputsEXT` and `EmitMeshTasksEXT` from
  every invocation (required); MSL calls `set_primitive_count` and
  `set_threadgroups_per_grid` from thread 0, Apple's documented pattern.
  Metal clip space has y up, so the MSL negates y once per vertex.
- Vertex-heavy size. `shape=vert` uses at most 128 vertices per workgroup
  because a `metal::mesh` is limited to 16 KB (hk allows 32 KB of mesh
  output); both sides use the same counts.
- Line rasterization. The coverage segment covers exactly one pixel under both
  the rectangle and the diamond-exit rule and never reaches the viewport edge,
  so no rule or clipping difference changes coverage; `--check` verifies it.
- Compiler output. The Vulkan row carries pipeline-executable statistics per
  stage; the Metal row carries the pipeline's thread limits and execution
  widths. `--dump-archive` and `--shader-stats` do not cover the mesh
  pipelines yet, so no compiled-code parity is claimed for render.mesh.
