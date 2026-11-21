// SPDX-License-Identifier: GPL-2.0-only
// Wave-1 90%-coverage Metal shaders: exact algorithmic twins of the GLSL
// shaders in suite/shaders/ (uber_*, conv2d_*, stereo, edge_*, deferred_*,
// mdi.*, pso_*).
#include <metal_stdlib>
using namespace metal;

/* ======================= compute.uber & concurrent.sram ======================= */

struct UberPC { uint seed; uint iters; };
struct UberIlpPC { uint seed; uint mask; };

#define MU_B(X, p, a, b, c, d) X(a, p) X(b, a) X(c, b) X(d, c)
#define MU_ALL(X) \
   MU_B(X, a23, a0, a1, a2, a3) \
   MU_B(X, a3,  a4, a5, a6, a7) \
   MU_B(X, a7,  a8, a9, a10, a11) \
   MU_B(X, a11, a12, a13, a14, a15) \
   MU_B(X, a15, a16, a17, a18, a19) \
   MU_B(X, a19, a20, a21, a22, a23)

#define MU_DECL(a, p) sd = sd * uint4(747796405u) + uint4(2891336453u); uint4 a = (sd ^ (sd >> uint4(15u))) + ks;
#define MU_STEP(a, p) a += p ^ (a >> uint4(9u));
#define MU_FOLD(a) sd = (sd ^ (a >> uint4(sd.x & 15u))) * uint4(16777619u);

#define MU_NARROW(ALT) { \
   MU_DECL(b0, b3) MU_DECL(b1, b0) MU_DECL(b2, b1) MU_DECL(b3, b2) \
   for (uint i = 0u; i < pc.iters; i++) { \
      if (ALT) { \
         b0 += b3 ^ (b0 << uint4(7u)); b1 += b0 ^ (b1 << uint4(7u)); \
         b2 += b1 ^ (b2 << uint4(7u)); b3 += b2 ^ (b3 << uint4(7u)); \
      } else { \
         b0 += b3 ^ (b0 >> uint4(9u)); b1 += b0 ^ (b1 >> uint4(9u)); \
         b2 += b1 ^ (b2 >> uint4(9u)); b3 += b2 ^ (b3 >> uint4(9u)); \
      } \
   } \
   MU_FOLD(b0) MU_FOLD(b1) MU_FOLD(b2) MU_FOLD(b3) \
}

kernel void uber_div_main(device uint *o [[buffer(0)]],
                          constant UberPC &pc [[buffer(1)]],
                          uint id [[thread_position_in_grid]])
{
   uint4 ks = uint4(pc.seed);
   uint4 sd = ((uint4(id) + uint4(1u, 2u, 3u, 4u)) * uint4(2654435761u)) ^ ks;
   MU_ALL(MU_DECL)
   for (uint s = 0u; s < 4u; s++) { MU_ALL(MU_STEP) }
   uint sel = ((id ^ (id >> 2u)) ^ pc.seed) & 3u;
   if (sel == 0u) {
      MU_FOLD(a0) MU_FOLD(a1) MU_FOLD(a2) MU_FOLD(a3) MU_FOLD(a4) MU_FOLD(a5)
      MU_NARROW(false)
      sd ^= uint4(0x243f6a88u);
   } else if (sel == 1u) {
      MU_FOLD(a6) MU_FOLD(a7) MU_FOLD(a8) MU_FOLD(a9) MU_FOLD(a10) MU_FOLD(a11)
      MU_NARROW(true)
      sd ^= uint4(0x85a308d3u);
   } else if (sel == 2u) {
      MU_FOLD(a12) MU_FOLD(a13) MU_FOLD(a14) MU_FOLD(a15) MU_FOLD(a16) MU_FOLD(a17)
      MU_NARROW(false)
      sd ^= uint4(0x13198a2eu);
   } else {
      MU_FOLD(a18) MU_FOLD(a19) MU_FOLD(a20) MU_FOLD(a21) MU_FOLD(a22) MU_FOLD(a23)
      MU_NARROW(true)
      sd ^= uint4(0x03707344u);
   }
   o[id] = sd.x ^ sd.y ^ sd.z ^ sd.w;
}

kernel void uber_uni_main(device uint *o [[buffer(0)]],
                          constant UberPC &pc [[buffer(1)]],
                          uint id [[thread_position_in_grid]],
                          uint grp [[threadgroup_position_in_grid]])
{
   uint4 ks = uint4(pc.seed);
   uint4 sd = ((uint4(id) + uint4(1u, 2u, 3u, 4u)) * uint4(2654435761u)) ^ ks;
   MU_ALL(MU_DECL)
   for (uint s = 0u; s < 4u; s++) { MU_ALL(MU_STEP) }
   uint sel = (grp ^ pc.seed) & 3u;
   if (sel == 0u) {
      MU_FOLD(a0) MU_FOLD(a1) MU_FOLD(a2) MU_FOLD(a3) MU_FOLD(a4) MU_FOLD(a5)
      MU_NARROW(false)
      sd ^= uint4(0x243f6a88u);
   } else if (sel == 1u) {
      MU_FOLD(a6) MU_FOLD(a7) MU_FOLD(a8) MU_FOLD(a9) MU_FOLD(a10) MU_FOLD(a11)
      MU_NARROW(true)
      sd ^= uint4(0x85a308d3u);
   } else if (sel == 2u) {
      MU_FOLD(a12) MU_FOLD(a13) MU_FOLD(a14) MU_FOLD(a15) MU_FOLD(a16) MU_FOLD(a17)
      MU_NARROW(false)
      sd ^= uint4(0x13198a2eu);
   } else {
      MU_FOLD(a18) MU_FOLD(a19) MU_FOLD(a20) MU_FOLD(a21) MU_FOLD(a22) MU_FOLD(a23)
      MU_NARROW(true)
      sd ^= uint4(0x03707344u);
   }
   o[id] = sd.x ^ sd.y ^ sd.z ^ sd.w;
}

kernel void uber_ilp_main(device const uint *tbl [[buffer(0)]],
                          device uint *o [[buffer(1)]],
                          constant UberIlpPC &pc [[buffer(2)]],
                          uint id [[thread_position_in_grid]])
{
   float4 a = float4(0.0f);
   float4 b = float4(0.0f);
   uint h = id * 2654435761u ^ pc.seed;
   for (uint it = 0u; it < 16u; it++) {
      uint base = id * 17u + it * 257u;
      uint t0 = tbl[(base + 0u * 4099u) & pc.mask];
      uint t1 = tbl[(base + 1u * 4099u) & pc.mask];
      uint t2 = tbl[(base + 2u * 4099u) & pc.mask];
      uint t3 = tbl[(base + 3u * 4099u) & pc.mask];
      uint t4 = tbl[(base + 4u * 4099u) & pc.mask];
      uint t5 = tbl[(base + 5u * 4099u) & pc.mask];
      uint t6 = tbl[(base + 6u * 4099u) & pc.mask];
      uint t7 = tbl[(base + 7u * 4099u) & pc.mask];
      a = fma(a, float4(1.0f), float4(float(t0), float(t1), float(t2), float(t3)) * 0.25f);
      b = fma(b, float4(1.0f), float4(float(t4), float(t5), float(t6), float(t7)) * 0.5f);
      h = (h ^ (t0 | (t1 << 4u) | (t2 << 8u) | (t3 << 12u) |
                (t4 << 16u) | (t5 << 20u) | (t6 << 24u) | (t7 << 28u))) * 16777619u;
   }
   uint4 r = uint4(a * 4.0f) + uint4(b * 2.0f);
   o[id] = (r.x | (r.y << 8u) | (r.z << 16u) | (r.w << 24u)) ^ h;
}

kernel void uber_sram_main(device uint *o [[buffer(0)]],
                           constant UberPC &pc [[buffer(1)]],
                           uint id [[thread_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]])
{
   threadgroup uint sh[4096];
   uint4 ks = uint4(pc.seed);
   uint4 sd = ((uint4(id) + uint4(1u, 2u, 3u, 4u)) * uint4(2654435761u)) ^ ks;
   for (uint j = 0u; j < 16u; j++)
      sh[lid * 16u + j] = sd.x ^ (j * 0x9e3779b9u);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   MU_ALL(MU_DECL)
   for (uint s = 0u; s < 4u; s++) { MU_ALL(MU_STEP) }
   uint sel = ((id ^ (id >> 2u)) ^ pc.seed) & 3u;
   if (sel == 0u) {
      MU_FOLD(a0) MU_FOLD(a1) MU_FOLD(a2) MU_FOLD(a3) MU_FOLD(a4) MU_FOLD(a5)
      MU_NARROW(false)
      sd ^= uint4(0x243f6a88u);
   } else if (sel == 1u) {
      MU_FOLD(a6) MU_FOLD(a7) MU_FOLD(a8) MU_FOLD(a9) MU_FOLD(a10) MU_FOLD(a11)
      MU_NARROW(true)
      sd ^= uint4(0x85a308d3u);
   } else if (sel == 2u) {
      MU_FOLD(a12) MU_FOLD(a13) MU_FOLD(a14) MU_FOLD(a15) MU_FOLD(a16) MU_FOLD(a17)
      MU_NARROW(false)
      sd ^= uint4(0x13198a2eu);
   } else {
      MU_FOLD(a18) MU_FOLD(a19) MU_FOLD(a20) MU_FOLD(a21) MU_FOLD(a22) MU_FOLD(a23)
      MU_NARROW(true)
      sd ^= uint4(0x03707344u);
   }
   sd.x ^= sh[((lid ^ 127u) * 16u) + (sd.x & 15u)];
   o[id] = sd.x ^ sd.y ^ sd.z ^ sd.w;
}

/* ======================= compute.conv2d ======================= */

struct ImgPC { int w; int h; };
constant uint kConvW17[17] = {
   16u, 86u, 342u, 1052u, 2554u, 4996u, 7914u, 10242u,
   11132u,
   10242u, 7914u, 4996u, 2554u, 1052u, 342u, 86u, 16u
};
constant uint kConvBilateral [[function_constant(10)]];

kernel void conv2d_h_main(device const uint *src [[buffer(0)]],
                          device uint *dst [[buffer(1)]],
                          constant ImgPC &pc [[buffer(2)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 gid [[threadgroup_position_in_grid]])
{
   threadgroup uint tile[16][32];
   int lx = int(tid.x), ly = int(tid.y);
   int gx = int(gid.x * 16u), gy = int(gid.y * 16u) + ly;
   int sy = clamp(gy, 0, pc.h - 1);
   for (int dx = lx; dx < 32; dx += 16) {
      int sx = clamp(gx + dx - 8, 0, pc.w - 1);
      tile[ly][dx] = src[sy * pc.w + sx];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int x = gx + lx;
   if (x >= pc.w || gy >= pc.h) return;

   uint cpx = tile[ly][lx + 8];
   uint4 cc = (uint4(cpx) >> uint4(0u, 8u, 16u, 24u)) & 255u;
   int cluma = int((cc.r + 2u * cc.g + cc.b) >> 2u);
   uint4 acc = uint4(0u);
   uint wsum = 0u;
   for (int t = 0; t < 17; t++) {
      uint p = tile[ly][lx + t];
      uint4 ch = (uint4(p) >> uint4(0u, 8u, 16u, 24u)) & 255u;
      uint wt = kConvW17[t];
      if (kConvBilateral != 0u) {
         int nluma = int((ch.r + 2u * ch.g + ch.b) >> 2u);
         int diff = abs(cluma - nluma);
         uint rw = diff < 64 ? (256u - uint(diff) * 4u) : 1u;
         wt = (wt >> 4u) * rw;
      }
      acc += ch * wt;
      wsum += wt;
   }
   uint4 res = (kConvBilateral == 0u) ? ((acc + uint4(32768u)) >> uint4(16u))
                                      : ((acc + uint4(wsum >> 1u)) / uint4(wsum));
   res = (res & 255u) << uint4(0u, 8u, 16u, 24u);
   dst[gy * pc.w + x] = res.r | res.g | res.b | res.a;
}

kernel void conv2d_v_main(device const uint *src [[buffer(0)]],
                          device uint *dst [[buffer(1)]],
                          constant ImgPC &pc [[buffer(2)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 gid [[threadgroup_position_in_grid]])
{
   threadgroup uint tile[32][16];
   int lx = int(tid.x), ly = int(tid.y);
   int gx = int(gid.x * 16u) + lx, gy = int(gid.y * 16u);
   int sx = clamp(gx, 0, pc.w - 1);
   for (int dy = ly; dy < 32; dy += 16) {
      int sy = clamp(gy + dy - 8, 0, pc.h - 1);
      tile[dy][lx] = src[sy * pc.w + sx];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int y = gy + ly;
   if (gx >= pc.w || y >= pc.h) return;

   uint cpx = tile[ly + 8][lx];
   uint4 cc = (uint4(cpx) >> uint4(0u, 8u, 16u, 24u)) & 255u;
   int cluma = int((cc.r + 2u * cc.g + cc.b) >> 2u);
   uint4 acc = uint4(0u);
   uint wsum = 0u;
   for (int t = 0; t < 17; t++) {
      uint p = tile[ly + t][lx];
      uint4 ch = (uint4(p) >> uint4(0u, 8u, 16u, 24u)) & 255u;
      uint wt = kConvW17[t];
      if (kConvBilateral != 0u) {
         int nluma = int((ch.r + 2u * ch.g + ch.b) >> 2u);
         int diff = abs(cluma - nluma);
         uint rw = diff < 64 ? (256u - uint(diff) * 4u) : 1u;
         wt = (wt >> 4u) * rw;
      }
      acc += ch * wt;
      wsum += wt;
   }
   uint4 res = (kConvBilateral == 0u) ? ((acc + uint4(32768u)) >> uint4(16u))
                                      : ((acc + uint4(wsum >> 1u)) / uint4(wsum));
   res = (res & 255u) << uint4(0u, 8u, 16u, 24u);
   dst[y * pc.w + gx] = res.r | res.g | res.b | res.a;
}

/* ======================= compute.stereo ======================= */

struct StereoPC { int w; int h; int pitch; int disp; };

static inline int load_gray(device const uint *buf, int pitch, int x, int y)
{
   uint off = uint(y * pitch + x);
   return int((buf[off >> 2u] >> ((off & 3u) * 8u)) & 255u);
}

kernel void stereo_main(device const uint *left_b [[buffer(0)]],
                        device const uint *right_b [[buffer(1)]],
                        device uint *out_b [[buffer(2)]],
                        constant StereoPC &pc [[buffer(3)]],
                        uint2 pos [[thread_position_in_grid]])
{
   int x = int(pos.x), y = int(pos.y);
   if (x >= pc.w || y >= pc.h) return;
   uint best_sad = 0x7fffffffu;
   uint best_d = 0u;
   for (int d = 0; d < pc.disp; d++) {
      uint sad = 0u;
      for (int dy = -5; dy <= 5; dy++) {
         int sy = clamp(y + dy, 0, pc.h - 1);
         for (int dx = -5; dx <= 5; dx++) {
            int lx = clamp(x + dx, 0, pc.w - 1);
            int rx = clamp(x + dx - d, 0, pc.w - 1);
            sad += uint(abs(load_gray(left_b, pc.pitch, lx, sy) -
                            load_gray(right_b, pc.pitch, rx, sy)));
         }
      }
      if (sad < best_sad) {
         best_sad = sad;
         best_d = uint(d);
      }
   }
   out_b[y * pc.pitch + x] = (best_d & 0xffu) | ((best_sad & 0xffffffu) << 8u);
}

/* ======================= compute.edge ======================= */

kernel void edge_sobel_main(device const uint *src [[buffer(0)]],
                            device uint *grad [[buffer(1)]],
                            device uint *mag [[buffer(2)]],
                            constant ImgPC &pc [[buffer(3)]],
                            uint2 pos [[thread_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 gid [[threadgroup_position_in_grid]],
                            uint lid [[thread_index_in_threadgroup]])
{
   threadgroup int tile[18][18];
   int bx = int(gid.x * 16u) - 1, by = int(gid.y * 16u) - 1;
   for (uint i = lid; i < 324u; i += 256u) {
      int tx = int(i % 18u), ty = int(i / 18u);
      int sx = clamp(bx + tx, 0, pc.w - 1), sy = clamp(by + ty, 0, pc.h - 1);
      tile[ty][tx] = int(src[sy * pc.w + sx]);
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int x = int(pos.x), y = int(pos.y);
   if (x >= pc.w || y >= pc.h) return;
   int lx = int(tid.x), ly = int(tid.y);
   int p00 = tile[ly + 0][lx + 0], p01 = tile[ly + 0][lx + 1], p02 = tile[ly + 0][lx + 2];
   int p10 = tile[ly + 1][lx + 0],                              p12 = tile[ly + 1][lx + 2];
   int p20 = tile[ly + 2][lx + 0], p21 = tile[ly + 2][lx + 1], p22 = tile[ly + 2][lx + 2];
   int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
   int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
   int idx = y * pc.w + x;
   grad[idx] = (uint(gx + 2048) & 0xfffu) | ((uint(gy + 2048) & 0xfffu) << 12u);
   mag[idx] = uint(abs(gx) + abs(gy));
}

kernel void edge_nms_main(device const uint *grad [[buffer(0)]],
                          device const uint *mag [[buffer(1)]],
                          device uint *dst [[buffer(2)]],
                          constant ImgPC &pc [[buffer(3)]],
                          uint2 pos [[thread_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 gid [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]])
{
   threadgroup uint tile[18][18];
   int bx = int(gid.x * 16u) - 1, by = int(gid.y * 16u) - 1;
   for (uint i = lid; i < 324u; i += 256u) {
      int tx = int(i % 18u), ty = int(i / 18u);
      int sx = clamp(bx + tx, 0, pc.w - 1), sy = clamp(by + ty, 0, pc.h - 1);
      tile[ty][tx] = mag[sy * pc.w + sx];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int x = int(pos.x), y = int(pos.y);
   if (x >= pc.w || y >= pc.h) return;
   int lx = int(tid.x), ly = int(tid.y);
   int idx = y * pc.w + x;
   uint g = grad[idx];
   int gx = int(g & 0xfffu) - 2048;
   int gy = int((g >> 12u) & 0xfffu) - 2048;
   int ax = abs(gx), ay = abs(gy);
   uint m = tile[ly + 1][lx + 1];
   uint n1, n2;
   if (ax >= 2 * ay) { n1 = tile[ly + 1][lx + 0]; n2 = tile[ly + 1][lx + 2]; }
   else if (ay >= 2 * ax) { n1 = tile[ly + 0][lx + 1]; n2 = tile[ly + 2][lx + 1]; }
   else if ((gx ^ gy) >= 0) { n1 = tile[ly + 0][lx + 0]; n2 = tile[ly + 2][lx + 2]; }
   else { n1 = tile[ly + 0][lx + 2]; n2 = tile[ly + 2][lx + 0]; }
   uint kept = (m >= n1 && m >= n2) ? m : 0u;
   dst[idx] = kept >= 180u ? 255u : (kept >= 90u ? 128u : 0u);
}

kernel void edge_harris_resp_main(device const uint *grad [[buffer(0)]],
                                  device int *resp [[buffer(1)]],
                                  constant ImgPC &pc [[buffer(2)]],
                                  uint2 pos [[thread_position_in_grid]],
                                  uint2 tid [[thread_position_in_threadgroup]],
                                  uint2 gid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_index_in_threadgroup]])
{
   threadgroup uint tile[18][18];
   int bx = int(gid.x * 16u) - 1, by = int(gid.y * 16u) - 1;
   for (uint i = lid; i < 324u; i += 256u) {
      int tx = int(i % 18u), ty = int(i / 18u);
      int sx = clamp(bx + tx, 0, pc.w - 1), sy = clamp(by + ty, 0, pc.h - 1);
      tile[ty][tx] = grad[sy * pc.w + sx];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int x = int(pos.x), y = int(pos.y);
   if (x >= pc.w || y >= pc.h) return;
   int lx = int(tid.x), ly = int(tid.y);
   int sxx = 0, syy = 0, sxy = 0;
   for (int dy = 0; dy < 3; dy++)
      for (int dx = 0; dx < 3; dx++) {
         uint g = tile[ly + dy][lx + dx];
         int gx = (int(g & 0xfffu) - 2048) >> 2;
         int gy = (int((g >> 12u) & 0xfffu) - 2048) >> 2;
         sxx += gx * gx; syy += gy * gy; sxy += gx * gy;
      }
   sxx >>= 4; syy >>= 4; sxy >>= 4;
   int det = sxx * syy - sxy * sxy;
   int tr = sxx + syy;
   int r = det - ((tr * tr) >> 4);
   resp[y * pc.w + x] = max(r, 0);
}

kernel void edge_harris_nms_main(device const int *resp [[buffer(0)]],
                                 device uint *dst [[buffer(1)]],
                                 device atomic_uint *counter [[buffer(2)]],
                                 constant ImgPC &pc [[buffer(3)]],
                                 uint2 pos [[thread_position_in_grid]],
                                 uint2 tid [[thread_position_in_threadgroup]],
                                 uint2 gid [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]])
{
   threadgroup int tile[18][18];
   int bx = int(gid.x * 16u) - 1, by = int(gid.y * 16u) - 1;
   for (uint i = lid; i < 324u; i += 256u) {
      int tx = int(i % 18u), ty = int(i / 18u);
      int sx = clamp(bx + tx, 0, pc.w - 1), sy = clamp(by + ty, 0, pc.h - 1);
      tile[ty][tx] = resp[sy * pc.w + sx];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   int x = int(pos.x), y = int(pos.y);
   if (x >= pc.w || y >= pc.h) return;
   int lx = int(tid.x), ly = int(tid.y);
   int c = tile[ly + 1][lx + 1];
   bool peak = c > 4096 &&
      c > tile[ly + 0][lx + 0] && c > tile[ly + 0][lx + 1] && c > tile[ly + 0][lx + 2] &&
      c > tile[ly + 1][lx + 0] && c >= tile[ly + 1][lx + 2] &&
      c >= tile[ly + 2][lx + 0] && c >= tile[ly + 2][lx + 1] && c >= tile[ly + 2][lx + 2];
   if (peak) {
      dst[y * pc.w + x] = 255u;
      atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
   } else {
      dst[y * pc.w + x] = 0u;
   }
}

/* ======================= render.deferred ======================= */

struct DefVertOut {
   float4 pos [[position]];
   uint qid [[flat]];
   float linz [[flat]];
};

struct DefGBuf {
   float4 g0 [[color(0)]];
   float4 g1 [[color(1)]];
   float4 g2 [[color(2)]];
   float  g3 [[color(3)]];
};

struct DefFinalOut {
   float4 col [[color(4)]];
};

vertex DefVertOut deferred_geom_vert(uint vid [[vertex_id]])
{
   uint quad = vid / 6u;
   uint corner = vid - quad * 6u;
   uint layer = 3u - (quad >> 10u);
   uint cell = quad & 1023u;
   uint qx = cell & 31u, qy = cell >> 5u;
   float2 off;
   if (corner == 0u) off = float2(0.0f, 0.0f);
   else if (corner == 1u) off = float2(1.0f, 0.0f);
   else if (corner == 2u) off = float2(0.0f, 1.0f);
   else if (corner == 3u) off = float2(1.0f, 0.0f);
   else if (corner == 4u) off = float2(1.0f, 1.0f);
   else off = float2(0.0f, 1.0f);
   float2 uv = (float2(float(qx), float(qy)) + off) / 32.0f;
   float z = 0.25f + float(layer) * 0.20f;
   /* Flip Y so pixel (0,0) is top-left, matching Vulkan and gb_wave1_ref.c. */
   DefVertOut o;
   o.pos = float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f, z, 1.0f);
   o.qid = cell + layer * 1024u;
   o.linz = z;
   return o;
}

fragment DefGBuf deferred_geom_frag(DefVertOut in [[stage_in]])
{
   uint qid = in.qid;
   float3 albedo = float3(
      0.2f + 0.7f * float((qid * 37u) & 255u) / 255.0f,
      0.2f + 0.7f * float((qid * 73u + 51u) & 255u) / 255.0f,
      0.2f + 0.7f * float((qid * 131u + 17u) & 255u) / 255.0f
   );
   float rough = 0.15f + 0.7f * float((qid * 19u) & 255u) / 255.0f;
   float3 n = normalize(float3(
      float((qid * 53u) & 255u) / 255.0f * 0.8f - 0.4f,
      float((qid * 97u) & 255u) / 255.0f * 0.8f - 0.4f,
      1.0f
   ));
   float metal = (qid & 3u) == 0u ? 0.8f : 0.1f;
   float3 emissive = float3(
      0.05f * float((qid * 11u) & 255u) / 255.0f,
      0.04f * float((qid * 23u) & 255u) / 255.0f,
      0.06f * float((qid * 41u) & 255u) / 255.0f
   );
   float ao = 0.6f + 0.4f * float((qid * 29u) & 255u) / 255.0f;
   DefGBuf g;
   g.g0 = float4(albedo, rough);
   g.g1 = float4(n, metal);
   g.g2 = float4(emissive, ao);
   g.g3 = in.linz;
   return g;
}

static inline float3 deferred_shade(float4 g0, float4 g1, float4 g2, float linz)
{
   constant float3 kLDir[4] = {
      float3( 0.57735027f,  0.57735027f, 0.57735027f),
      float3(-0.53452248f,  0.26726124f, 0.80178373f),
      float3( 0.26726124f, -0.53452248f, 0.80178373f),
      float3(-0.40824829f, -0.40824829f, 0.81649658f)
   };
   constant float3 kLCol[4] = {
      float3(0.95f, 0.90f, 0.85f),
      float3(0.40f, 0.55f, 0.85f),
      float3(0.75f, 0.45f, 0.30f),
      float3(0.35f, 0.70f, 0.50f)
   };
   float3 albedo = g0.rgb;
   float rough = g0.a;
   float3 n = normalize(g1.xyz);
   float metal = g1.w;
   float3 emissive = g2.rgb;
   float ao = g2.a;
   float3 col = albedo * 0.12f * ao + emissive;
   float shin = 8.0f + (1.0f - rough) * 24.0f;
   float atten = 1.0f / (1.0f + linz * linz);
   for (int i = 0; i < 4; i++) {
      float ndl = max(dot(n, kLDir[i]), 0.0f);
      float3 h = normalize(kLDir[i] + float3(0.0f, 0.0f, 1.0f));
      float ndh = max(dot(n, h), 0.0f);
      float spec = pow(ndh, shin) * (0.2f + 0.8f * metal);
      float diff = ndl * (1.0f - 0.5f * metal);
      col += (albedo * diff + float3(spec)) * kLCol[i] * atten;
   }
   return clamp(col, 0.0f, 1.0f);
}

/* On-chip TBDR tile-memory lighting pass (programmable blending [[color(0..3)]]). */
fragment DefFinalOut deferred_tile_frag(DefGBuf g)
{
   DefFinalOut o;
   o.col = float4(deferred_shade(g.g0, g.g1, g.g2, g.g3), 1.0f);
   return o;
}

/* On-chip TBDR tile-memory lighting pass (4x MSAA, forced per-sample rate via [[sample_id]]). */
fragment DefFinalOut deferred_tile_msaa_frag(DefGBuf g, uint sid [[sample_id]])
{
   (void)sid;
   DefFinalOut o;
   o.col = float4(deferred_shade(g.g0, g.g1, g.g2, g.g3), 1.0f);
   return o;
}

/* Forced DRAM spill lighting pass (1x). */
fragment float4 deferred_dram_frag(float4 pos [[position]],
                                   texture2d<float> t0 [[texture(0)]],
                                   texture2d<float> t1 [[texture(1)]],
                                   texture2d<float> t2 [[texture(2)]],
                                   texture2d<float> t3 [[texture(3)]])
{
   uint2 p = uint2(pos.xy);
   return float4(deferred_shade(t0.read(p), t1.read(p), t2.read(p), t3.read(p).r), 1.0f);
}

/* Forced DRAM spill lighting pass (4x MSAA). */
fragment float4 deferred_dram_msaa_frag(float4 pos [[position]],
                                        texture2d_ms<float> t0 [[texture(0)]],
                                        texture2d_ms<float> t1 [[texture(1)]],
                                        texture2d_ms<float> t2 [[texture(2)]],
                                        texture2d_ms<float> t3 [[texture(3)]])
{
   uint2 p = uint2(pos.xy);
   float3 total = float3(0.0f);
   for (uint s = 0; s < 4; s++)
      total += deferred_shade(t0.read(p, s), t1.read(p, s), t2.read(p, s), t3.read(p, s).r);
   return float4(total * 0.25f, 1.0f);
}

/* ======================= render.mdi ======================= */

struct MdiVertIn { float2 pos [[attribute(0)]]; };
struct MdiVertOut {
   float4 pos [[position]];
   uint tex_idx [[flat]];
   float2 uv;
};
struct MdiPC { uint bindless; };
struct MdiTexArg {
   array<texture2d<float>, 1024> textures [[id(0)]];
};

vertex MdiVertOut mdi_vert(MdiVertIn in [[stage_in]],
                           constant MdiPC &pc [[buffer(1)]],
                           uint iid [[instance_id]])
{
   uint draw_id = iid;
   uint cell = draw_id % 10000u;
   uint cx = cell % 100u, cy = cell / 100u;
   float2 px = float2(float(cx * 10u), float(cy * 10u)) + in.pos * 10.0f;
   MdiVertOut o;
   o.pos = float4((px.x / 1024.0f) * 2.0f - 1.0f, 1.0f - (px.y / 1024.0f) * 2.0f, 0.0f, 1.0f);
   o.tex_idx = draw_id % pc.bindless;
   o.uv = in.pos;
   return o;
}

fragment float4 mdi_frag(MdiVertOut in [[stage_in]],
                         constant MdiTexArg &arg [[buffer(0)]])
{
   constexpr sampler s(mag_filter::nearest, min_filter::nearest);
   return arg.textures[in.tex_idx].sample(s, in.uv);
}

/* ======================= compile.pso ======================= */

constant uint kPsoVarId [[function_constant(20)]];
constant uint kPsoLoopN [[function_constant(21)]];
constant uint kPsoMask  [[function_constant(22)]];
constant uint kPsoNonce [[function_constant(23)]];

kernel void pso_comp_main(device uint *o [[buffer(0)]],
                          uint id [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]])
{
   threadgroup uint scratch[128];
   uint acc = (id * 2654435761u) ^ kPsoNonce ^ (kPsoVarId * 0x9e3779b9u);
   scratch[lid] = acc;
   scratch[lid + 64u] = acc ^ 0xa5a5a5a5u;
   threadgroup_barrier(mem_flags::mem_threadgroup);
   for (uint i = 0u; i < kPsoLoopN; i++) {
      uint s = scratch[(lid + i + (kPsoVarId & 7u)) & 127u];
      if (((i + kPsoVarId) & kPsoMask) == 0u)
         acc = (acc * 16777619u) ^ (s >> ((kPsoVarId & 7u) + 1u));
      else
         acc = (acc + s + kPsoNonce) ^ (acc << ((kPsoVarId & 3u) + 1u));
   }
   o[id] = acc;
}

kernel void pso_uber_main(device uint *o [[buffer(0)]],
                          uint id [[thread_position_in_grid]])
{
   uint4 v0 = uint4(id, id + 1u, id + 2u, id + 3u) ^ uint4(kPsoNonce);
   uint4 v1 = v0 * uint4(747796405u) + uint4(kPsoVarId);
   uint4 v2 = v1 ^ (v0 >> uint4(5u));
   uint4 v3 = v2 * uint4(2654435761u) + uint4(kPsoNonce);
   uint4 v4 = v3 ^ (v1 << uint4(3u));
   uint4 v5 = v4 + v2 * uint4(16777619u);
   uint4 v6 = v5 ^ (v3 >> uint4(7u));
   uint4 v7 = v6 + v4 ^ uint4(kPsoVarId * 0x9e3779b9u);
   for (uint i = 0u; i < kPsoLoopN; i++) {
      uint b = (id + i + kPsoVarId) & kPsoMask;
      if (b == 0u) { v0 += v1 ^ (v2 >> uint4(3u)); v3 = v3 * uint4(16777619u) ^ v0; }
      else if (b == 1u) { v1 += v2 ^ (v3 << uint4(5u)); v4 = v4 * uint4(747796405u) + v1; }
      else if (b == 2u) { v2 += v4 ^ (v5 >> uint4(7u)); v5 ^= v6 + uint4(kPsoNonce); }
      else if (b == 3u) { v3 += v5 ^ (v6 << uint4(2u)); v6 = (v6 ^ v7) * uint4(2654435761u); }
      else { v4 += v6 ^ (v7 >> uint4(9u)); v7 = v7 + v0 + uint4(kPsoVarId ^ kPsoNonce); }
   }
   uint4 f = (v0 ^ v1) + (v2 ^ v3) + (v4 ^ v5) + (v6 ^ v7);
   o[id] = f.x ^ f.y ^ f.z ^ f.w;
}

struct PsoVertOut { float4 pos [[position]]; float4 uv; };

vertex PsoVertOut pso_gfx_vert(uint vid [[vertex_id]])
{
   float2 p = float2(float((vid << 1) & 2), float(vid & 2));
   float s = float((kPsoVarId ^ kPsoNonce) & 255u) / 255.0f;
   PsoVertOut o;
   o.pos = float4(p * 2.0f - 1.0f, 0.5f, 1.0f);
   o.uv = float4(p, s, 1.0f - s);
   return o;
}

fragment float4 pso_gfx_frag(PsoVertOut in [[stage_in]])
{
   float3 n = normalize(float3(in.uv.xy * 2.0f - 1.0f, 0.8f + 0.1f * in.uv.z));
   float3 albedo = float3(in.uv.x, in.uv.y, in.uv.z);
   float3 col = albedo * 0.1f;
   float phase = float((kPsoNonce ^ (kPsoVarId * 17u)) & 1023u) * 0.006f;
   for (uint i = 0u; i < kPsoLoopN; i++) {
      float a = phase + float(i) * 0.7f;
      float3 l = normalize(float3(sin(a), cos(a), 0.7f));
      float ndl = max(dot(n, l), 0.0f);
      float3 h = normalize(l + float3(0.0f, 0.0f, 1.0f));
      float spec = pow(max(dot(n, h), 0.0f), 8.0f + float((i + kPsoVarId) & kPsoMask) * 6.0f);
      col += (albedo * ndl + float3(spec * 0.3f)) * (0.25f / float(kPsoLoopN));
   }
   return float4(clamp(col, 0.0f, 1.0f), 1.0f);
}

/* ======================= compress.bw ======================= */

struct CompressPC { float4 param; };
struct CompressStorePC { int w; int h; float k; float pad; };

fragment DefGBuf compress_mrt4_frag(float4 pos [[position]],
                                    constant CompressPC &pc [[buffer(0)]])
{
   float2 p = floor(pos.xy * 0.0625f) * 0.015625f;
   float k = pc.param.x;
   DefGBuf g;
   g.g0 = float4(fract(p.x + k), fract(p.y + 0.25f), fract(p.x + p.y), 1.0f);
   g.g1 = float4(fract(p.y + k), fract(p.x + 0.50f), 0.75f, 0.5f);
   g.g2 = float4(0.25f, fract(p.x * 0.5f + k), fract(p.y * 0.5f), 1.0f);
   g.g3 = 0.25f + 0.5f * fract(p.x + k);
   return g;
}

fragment float4 compress_write1_frag(float4 pos [[position]],
                                     constant CompressPC &pc [[buffer(0)]])
{
   float2 p = floor(pos.xy * 0.0625f) * 0.015625f;
   float k = pc.param.x;
   return float4(fract(p.x + k), fract(p.y + pc.param.y), fract(p.x + p.y + pc.param.z), 1.0f);
}

fragment float4 compress_sample4_frag(float4 pos [[position]],
                                      texture2d<float> t0 [[texture(0)]],
                                      texture2d<float> t1 [[texture(1)]],
                                      texture2d<float> t2 [[texture(2)]],
                                      texture2d<float> t3 [[texture(3)]])
{
   uint2 p = uint2(pos.xy);
   float4 a = t0.read(p);
   float4 b = t1.read(p);
   float4 c = t2.read(p);
   float d = t3.read(p).r;
   return float4(clamp((a.rgb + b.rgb + c.rgb) * 0.25f + float3(d * 0.25f), 0.0f, 1.0f), 1.0f);
}

fragment float4 compress_sample1_frag(float4 pos [[position]],
                                      texture2d<float> t0 [[texture(0)]])
{
   uint2 p = uint2(pos.xy);
   return float4(clamp(t0.read(p).rgb, 0.0f, 1.0f), 1.0f);
}

fragment float4 compress_array_frag(float4 pos [[position]],
                                    texture2d_array<float> t_arr [[texture(0)]])
{
   uint2 p = uint2(pos.xy);
   float3 acc = float3(0.0f);
   for (uint l = 0; l < 6; l++)
      acc += t_arr.read(p, l).rgb;
   return float4(clamp(acc * (1.0f / 6.0f), 0.0f, 1.0f), 1.0f);
}

kernel void compress_store_main(texture2d<float, access::write> dst [[texture(0)]],
                                constant CompressStorePC &pc [[buffer(0)]],
                                uint2 pos [[thread_position_in_grid]])
{
   if (int(pos.x) >= pc.w || int(pos.y) >= pc.h) return;
   float2 q = floor(float2(pos) * 0.0625f) * 0.015625f;
   dst.write(float4(fract(q.x + pc.k), fract(q.y + 0.25f), fract(q.x + q.y), 1.0f), pos);
}
