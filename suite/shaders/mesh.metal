// SPDX-License-Identifier: GPL-2.0-only
// render.mesh: the same algorithms as mesh.mesh, mesh.task and mesh.frag.
// Not part of the suite's main library: gpubench-mtl compiles this file once
// per configuration with preprocessor macros (tools/build-macos.sh embeds it
// as gb_msl_mesh_source):
//   WG, TOPO_LINE, SHAPE_VERT, PERPRIM, VIEWS, TASK  as in mesh.mesh
//   VIEW_AMP   views=2: 1 = vertex amplification ([[amplification_id]] gives
//              the view), 0 = twice the mesh groups, each primitive routed by
//              [[render_target_array_index]]
//   LAYER_OUT  1: the primitive data carries [[render_target_array_index]]
//              (0 with VIEW_AMP, the view mapping adds the layer offset)
// Functions: mesh_object (TASK only), mesh_main, mesh_frag.
#include <metal_stdlib>
using namespace metal;

#define CELL 4u // GB_MESH_CELL: pixels per grid cell
#define GW (WG == 32 ? 8u : 16u)
#define GH (uint(WG) / GW)

#if TOPO_LINE
#define NV (2 * WG)
#define NP WG
#define TOPO topology::line
#elif SHAPE_VERT
#if WG == 32
#define NV 64
#else
#define NV 128
#endif
#define NP (NV / 4)
#define TOPO topology::triangle
#else
#define NV WG
#define NP (2 * WG)
#define TOPO topology::triangle
#endif

#define HEAVY (SHAPE_VERT && !TOPO_LINE)
#define HAVE_PRIM (PERPRIM || LAYER_OUT)

struct PC {
   uint tiles_x;     // tiles per row
   uint tiles_total; // tiles in the target
   uint res;         // target width = height
   uint npix;        // res * res
};

struct Payload {
   uint base;
};

struct VOut {
   float4 pos [[position]];
   float4 v_color [[flat]];
#if HEAVY
   float4 v_e0;
   float4 v_e1;
   float4 v_e2;
   float4 v_e3;
#endif
};

#if HAVE_PRIM
struct POut {
#if PERPRIM
   float4 p_color [[flat]];
   uint p_id [[flat]];
#endif
#if LAYER_OUT
   uint layer [[render_target_array_index]];
#endif
};
using MeshT = metal::mesh<VOut, POut, NV, NP, TOPO>;
#else
using MeshT = metal::mesh<VOut, void, NV, NP, TOPO>;
#endif

// lowbias32 integer hash; colour bytes = h & 0xff, (h >> 8) & 0xff, (h >> 16) & 0xff, 255.
static uint gb_hash(uint x)
{
   x ^= x >> 16u;
   x *= 0x7feb352du;
   x ^= x >> 15u;
   x *= 0x846ca68bu;
   x ^= x >> 16u;
   return x;
}

static float4 gb_color(uint g, uint view)
{
   uint h = gb_hash(g * 2u + view);
   return float4(float(h & 255u), float((h >> 8u) & 255u), float((h >> 16u) & 255u), 255.0f) / 255.0f;
}

// Pixel coordinates (origin top-left, y down) to clip space. Metal's clip y
// points up (Vulkan's down), hence the sign: the same pixels on both sides.
static float4 gb_clip(float2 px, uint res)
{
   float2 n = px * (2.0f / float(res)) - 1.0f;
   return float4(n.x, -n.y, 0.0f, 1.0f);
}

#if TASK
// stage=task: each object threadgroup of 32 threads launches 32 mesh
// threadgroups and passes the index of the first one in the payload.
[[object]] void mesh_object(object_data Payload &payload [[payload]],
                            mesh_grid_properties mgp,
                            uint t [[thread_index_in_threadgroup]],
                            uint tg [[threadgroup_position_in_grid]])
{
   if (t == 0u) {
      payload.base = tg * 32u;
      mgp.set_threadgroups_per_grid(uint3(32u, 1u, 1u));
   }
}
#endif

[[mesh]] void mesh_main(MeshT out,
#if TASK
                        const object_data Payload &payload [[payload]],
#endif
                        constant PC &pc [[buffer(0)]],
#if VIEWS > 1 && !VIEW_AMP
                        constant uint &groups [[buffer(1)]],
#endif
#if VIEWS > 1 && VIEW_AMP
                        uint amp [[amplification_id]],
#endif
                        uint t [[thread_index_in_threadgroup]],
                        uint gid [[threadgroup_position_in_grid]])
{
#if TASK
   uint lin = payload.base + gid;
#else
   uint lin = gid;
#endif
#if VIEWS > 1 && VIEW_AMP
   uint g = lin, view = amp;
#elif VIEWS > 1
   uint g = lin % groups, view = lin / groups;
#else
   uint g = lin, view = 0u;
#endif
   float4 c = gb_color(g, view);
   if (t == 0u)
      out.set_primitive_count(NP);
#if HAVE_PRIM
   POut pr;
#if LAYER_OUT && VIEWS > 1 && !VIEW_AMP
   pr.layer = view;
#elif LAYER_OUT
   pr.layer = 0u;
#endif
#endif

#if TOPO_LINE
   // A 1-pixel segment over pixel x of row y at the row's centre (see mesh.mesh).
   uint i = (g * uint(WG) + t) % pc.npix;
   uint xi = i % pc.res;
   float x = float(xi), y = float(i / pc.res) + 0.5f;
   bool last = xi == pc.res - 1u;
   float x0 = last ? x + 1.0f - 1.0f / 64.0f : x + 1.0f / 64.0f;
   float x1 = last ? x - 1.0f / 64.0f : x + 1.0f + 1.0f / 64.0f;
   VOut v;
   v.pos = gb_clip(float2(x0, y), pc.res);
   v.v_color = c;
   out.set_vertex(t, v);
   v.pos = gb_clip(float2(x1, y), pc.res);
   out.set_vertex(t + uint(WG), v);
   out.set_index(2u * t, uchar(t));
   out.set_index(2u * t + 1u, uchar(t + uint(WG)));
#if HAVE_PRIM
#if PERPRIM
   pr.p_color = c;
   pr.p_id = g * uint(NP) + t;
#endif
   out.set_primitive(t, pr);
#endif
#else
   uint tile = g % pc.tiles_total;
   float2 org = float2(float((tile % pc.tiles_x) * GW * CELL), float((tile / pc.tiles_x) * GH * CELL));
#if SHAPE_VERT
   for (uint vi = t; vi < uint(NV); vi += uint(WG)) {
      uint cell = 2u * (vi >> 2u), corner = vi & 3u;
      float2 px = org + float2(float(cell % GW + (corner & 1u)), float(cell / GW + (corner >> 1u))) * float(CELL);
      VOut v;
      v.pos = gb_clip(px, pc.res);
      v.v_color = c;
      float4 b = float4(px, float(vi), float(g & 1023u)) / 1024.0f;
      v.v_e0 = b;
      v.v_e1 = b.yzwx + c;
      v.v_e2 = b.zwxy * c;
      v.v_e3 = b.wxyz - c;
      out.set_vertex(vi, v);
   }
   if (t < uint(NP)) {
      out.set_index(3u * t, uchar(4u * t));
      out.set_index(3u * t + 1u, uchar(4u * t + 1u));
      out.set_index(3u * t + 2u, uchar(4u * t + 2u));
#if HAVE_PRIM
#if PERPRIM
      pr.p_color = c;
      pr.p_id = g * uint(NP) + t;
#endif
      out.set_primitive(t, pr);
#endif
   }
#else
   uint gi = t % GW, gj = t / GW;
   VOut v;
   v.pos = gb_clip(org + float2(float(gi), float(gj)) * float(CELL), pc.res);
   v.v_color = c;
   out.set_vertex(t, v);
   // Thread t splits the grid quad whose top-left vertex it owns; threads on
   // the last column or row split the neighbouring quad along its other
   // diagonal (see mesh.mesh).
   uint qi = min(gi, GW - 2u), qj = min(gj, GH - 2u);
   uint a = qj * GW + qi, b = a + 1u, d = a + GW, e = d + 1u;
   bool alt = gi == GW - 1u || gj == GH - 1u;
   uint3 t0 = alt ? uint3(a, b, d) : uint3(a, b, e);
   uint3 t1 = alt ? uint3(b, e, d) : uint3(a, e, d);
   out.set_index(6u * t, uchar(t0.x));
   out.set_index(6u * t + 1u, uchar(t0.y));
   out.set_index(6u * t + 2u, uchar(t0.z));
   out.set_index(6u * t + 3u, uchar(t1.x));
   out.set_index(6u * t + 4u, uchar(t1.y));
   out.set_index(6u * t + 5u, uchar(t1.z));
#if HAVE_PRIM
#if PERPRIM
   pr.p_color = c;
   pr.p_id = g * uint(NP) + 2u * t;
#endif
   out.set_primitive(2u * t, pr);
#if PERPRIM
   pr.p_id = g * uint(NP) + 2u * t + 1u;
#endif
   out.set_primitive(2u * t + 1u, pr);
#endif
#endif
#endif
}

struct FIn {
   VOut v;
#if HAVE_PRIM
   POut p;
#endif
};

fragment float4 mesh_frag(FIn in [[stage_in]])
{
   float4 c = in.v.v_color;
#if HEAVY
   c += (in.v.v_e0 + in.v.v_e1 + in.v.v_e2 + in.v.v_e3) * 0.0625f;
#endif
#if PERPRIM
   // p_id < 2^31 always; the select keeps the id live.
   c = in.p.p_id < 0x80000000u ? mix(c, in.p.p_color, 0.5f) : float4(0.0f);
#endif
   return c;
}
