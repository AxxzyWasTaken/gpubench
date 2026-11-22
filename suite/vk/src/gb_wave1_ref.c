// SPDX-License-Identifier: GPL-2.0-only
#include "gb_wave1_ref.h"
#include "gb_params.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * compute.uber CPU reference
 * ========================================================================= */

typedef struct { uint32_t x, y, z, w; } u4;

static inline u4 u4_splat(uint32_t v) { return (u4){v, v, v, v}; }
static inline u4 u4_add(u4 a, u4 b) { return (u4){a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
static inline u4 u4_mul(u4 a, u4 b) { return (u4){a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
static inline u4 u4_xor(u4 a, u4 b) { return (u4){a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w}; }
static inline u4 u4_shr(u4 a, uint32_t s) { return (u4){a.x >> s, a.y >> s, a.z >> s, a.w >> s}; }
static inline u4 u4_shl(u4 a, uint32_t s) { return (u4){a.x << s, a.y << s, a.z << s, a.w << s}; }

static inline u4
uber_decl(u4 *sd, u4 ks)
{
   *sd = u4_add(u4_mul(*sd, u4_splat(747796405u)), u4_splat(2891336453u));
   return u4_add(u4_xor(*sd, u4_shr(*sd, 15u)), ks);
}

static inline void
uber_fold(u4 *sd, u4 a)
{
   uint32_t sh = sd->x & 15u;
   *sd = u4_mul(u4_xor(*sd, u4_shr(a, sh)), u4_splat(16777619u));
}

static inline void
uber_narrow_loop(u4 *sd, u4 ks, uint32_t iters, bool alt)
{
   u4 b[4];
   for (int i = 0; i < 4; i++)
      b[i] = uber_decl(sd, ks);
   for (uint32_t it = 0; it < iters; it++) {
      for (int i = 0; i < 4; i++) {
         u4 p = b[(i + 3) & 3];
         u4 t = alt ? u4_shl(b[i], 7u) : u4_shr(b[i], 9u);
         b[i] = u4_add(b[i], u4_xor(p, t));
      }
   }
   for (int i = 0; i < 4; i++)
      uber_fold(sd, b[i]);
}

void
gb_uber_fill_ilp_input(uint32_t *buf, size_t nwords)
{
   uint32_t s = 0x243f6a88u;
   for (size_t i = 0; i < nwords; i++) {
      s = s * 1664525u + 1013904223u;
      /* Small integers 0..15: exact in both IEEE binary16 and binary32. */
      buf[i] = (s >> 28u) & 15u;
   }
}

uint32_t
gb_uber_ref(enum gb_uber_kind kind, const uint32_t *ilp_in, uint32_t *out, uint32_t nthreads)
{
   const uint32_t seed = GB_UBER_SEED;
   if (kind == GB_UBER_ILP) {
      const uint32_t mask = GB_UBER_ILP_WORDS - 1u;
      for (uint32_t id = 0; id < nthreads; id++) {
         float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
         float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
         uint32_t h = id * 2654435761u ^ seed;
         for (uint32_t it = 0; it < 16u; it++) {
            uint32_t base = id * 17u + it * 257u;
            uint32_t t0 = ilp_in[(base + 0u * 4099u) & mask];
            uint32_t t1 = ilp_in[(base + 1u * 4099u) & mask];
            uint32_t t2 = ilp_in[(base + 2u * 4099u) & mask];
            uint32_t t3 = ilp_in[(base + 3u * 4099u) & mask];
            uint32_t t4 = ilp_in[(base + 4u * 4099u) & mask];
            uint32_t t5 = ilp_in[(base + 5u * 4099u) & mask];
            uint32_t t6 = ilp_in[(base + 6u * 4099u) & mask];
            uint32_t t7 = ilp_in[(base + 7u * 4099u) & mask];
            /* Exact integer/half-integer FMA chains (zero rounding error). */
            a0 = a0 * 1.0f + (float)t0 * 0.25f;
            a1 = a1 * 1.0f + (float)t1 * 0.25f;
            a2 = a2 * 1.0f + (float)t2 * 0.25f;
            a3 = a3 * 1.0f + (float)t3 * 0.25f;
            b0 = b0 * 1.0f + (float)t4 * 0.5f;
            b1 = b1 * 1.0f + (float)t5 * 0.5f;
            b2 = b2 * 1.0f + (float)t6 * 0.5f;
            b3 = b3 * 1.0f + (float)t7 * 0.5f;
            h = (h ^ (t0 | (t1 << 4u) | (t2 << 8u) | (t3 << 12u) |
                      (t4 << 16u) | (t5 << 20u) | (t6 << 24u) | (t7 << 28u))) * 16777619u;
         }
         uint32_t r0 = (uint32_t)(a0 * 4.0f) + (uint32_t)(b0 * 2.0f);
         uint32_t r1 = (uint32_t)(a1 * 4.0f) + (uint32_t)(b1 * 2.0f);
         uint32_t r2 = (uint32_t)(a2 * 4.0f) + (uint32_t)(b2 * 2.0f);
         uint32_t r3 = (uint32_t)(a3 * 4.0f) + (uint32_t)(b3 * 2.0f);
         out[id] = (r0 | (r1 << 8u) | (r2 << 16u) | (r3 << 24u)) ^ h;
      }
      return gb_crc32(out, (size_t)nthreads * sizeof(uint32_t));
   }

   /* GB_UBER_DIV, GB_UBER_UNI, GB_UBER_SRAM */
   uint32_t *sh = kind == GB_UBER_SRAM ? malloc(GB_UBER_LOCAL * 16u * sizeof(uint32_t)) : NULL;
   const uint32_t ngroups = nthreads / GB_UBER_LOCAL;
   const u4 ks = u4_splat(seed);

   for (uint32_t grp = 0; grp < ngroups; grp++) {
      if (kind == GB_UBER_SRAM) {
         for (uint32_t lid = 0; lid < GB_UBER_LOCAL; lid++) {
            uint32_t id = grp * GB_UBER_LOCAL + lid;
            uint32_t sx = ((id + 1u) * 2654435761u) ^ seed;
            for (uint32_t j = 0; j < 16u; j++)
               sh[lid * 16u + j] = sx ^ (j * 0x9e3779b9u);
         }
      }
      for (uint32_t lid = 0; lid < GB_UBER_LOCAL; lid++) {
         uint32_t id = grp * GB_UBER_LOCAL + lid;
         u4 sd = {
            ((id + 1u) * 2654435761u) ^ seed,
            ((id + 2u) * 2654435761u) ^ seed,
            ((id + 3u) * 2654435761u) ^ seed,
            ((id + 4u) * 2654435761u) ^ seed,
         };
         u4 a[24];
         for (int i = 0; i < 24; i++)
            a[i] = uber_decl(&sd, ks);
         for (int step = 0; step < 4; step++) {
            for (int i = 0; i < 24; i++) {
               u4 p = a[i == 0 ? 23 : i - 1];
               a[i] = u4_add(a[i], u4_xor(p, u4_shr(a[i], 9u)));
            }
         }
         uint32_t sel = (kind == GB_UBER_UNI) ? ((grp ^ seed) & 3u) : (((id ^ (id >> 2u)) ^ seed) & 3u);
         if (sel == 0u) {
            for (int i = 0; i < 6; i++) uber_fold(&sd, a[i]);
            uber_narrow_loop(&sd, ks, GB_UBER_ITERS, false);
            sd = u4_xor(sd, u4_splat(0x243f6a88u));
         } else if (sel == 1u) {
            for (int i = 6; i < 12; i++) uber_fold(&sd, a[i]);
            uber_narrow_loop(&sd, ks, GB_UBER_ITERS, true);
            sd = u4_xor(sd, u4_splat(0x85a308d3u));
         } else if (sel == 2u) {
            for (int i = 12; i < 18; i++) uber_fold(&sd, a[i]);
            uber_narrow_loop(&sd, ks, GB_UBER_ITERS, false);
            sd = u4_xor(sd, u4_splat(0x13198a2eu));
         } else {
            for (int i = 18; i < 24; i++) uber_fold(&sd, a[i]);
            uber_narrow_loop(&sd, ks, GB_UBER_ITERS, true);
            sd = u4_xor(sd, u4_splat(0x03707344u));
         }
         if (kind == GB_UBER_SRAM)
            sd.x ^= sh[((lid ^ 127u) * 16u) + (sd.x & 15u)];
         out[id] = sd.x ^ sd.y ^ sd.z ^ sd.w;
      }
   }
   free(sh);
   return gb_crc32(out, (size_t)nthreads * sizeof(uint32_t));
}

/* =========================================================================
 * compute.conv2d CPU reference (17-tap separable Gaussian / Bilateral)
 * ========================================================================= */

static const uint32_t CONV_W[17] = {
   16u, 86u, 342u, 1052u, 2554u, 4996u, 7914u, 10242u,
   11132u,
   10242u, 7914u, 4996u, 2554u, 1052u, 342u, 86u, 16u,
};

void
gb_conv2d_fill_input(uint32_t *img, uint32_t w, uint32_t h)
{
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         uint32_t idx = y * w + x;
         uint32_t z = idx * 2654435761u ^ (x * 73856093u) ^ (y * 19349663u);
         uint32_t r = ((x & 63u) < 32u ? 200u : 40u) ^ ((z >> 0u) & 31u);
         uint32_t g = ((y & 63u) < 32u ? 180u : 55u) ^ ((z >> 8u) & 31u);
         uint32_t b = (((x + y) & 31u) * 7u) ^ ((z >> 16u) & 15u);
         img[idx] = (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | (255u << 24u);
      }
   }
}

static inline uint32_t
conv_pixel_1d(const uint32_t *src, int x, int y, int w, int h, bool vert, bool bilateral)
{
   uint32_t cpx = src[(uint32_t)y * (uint32_t)w + (uint32_t)x];
   uint32_t cr = cpx & 255u, cg = (cpx >> 8u) & 255u, cb = (cpx >> 16u) & 255u;
   int cluma = (int)((cr + 2u * cg + cb) >> 2u);
   uint32_t ar = 0, ag = 0, ab = 0, aa = 0, wsum = 0;
   for (int t = -8; t <= 8; t++) {
      int sx = vert ? x : (x + t < 0 ? 0 : (x + t >= w ? w - 1 : x + t));
      int sy = vert ? (y + t < 0 ? 0 : (y + t >= h ? h - 1 : y + t)) : y;
      uint32_t p = src[(uint32_t)sy * (uint32_t)w + (uint32_t)sx];
      uint32_t pr = p & 255u, pg = (p >> 8u) & 255u, pb = (p >> 16u) & 255u, pa = (p >> 24u) & 255u;
      uint32_t wt = CONV_W[t + 8];
      if (bilateral) {
         int nluma = (int)((pr + 2u * pg + pb) >> 2u);
         int diff = cluma > nluma ? cluma - nluma : nluma - cluma;
         uint32_t rw = diff < 64 ? (256u - (uint32_t)diff * 4u) : 1u;
         wt = (wt >> 4u) * rw;
      }
      ar += pr * wt;
      ag += pg * wt;
      ab += pb * wt;
      aa += pa * wt;
      wsum += wt;
   }
   uint32_t r, g, b, a;
   if (!bilateral) {
      r = (ar + 32768u) >> 16u;
      g = (ag + 32768u) >> 16u;
      b = (ab + 32768u) >> 16u;
      a = (aa + 32768u) >> 16u;
   } else {
      uint32_t half = wsum >> 1u;
      r = (ar + half) / wsum;
      g = (ag + half) / wsum;
      b = (ab + half) / wsum;
      a = (aa + half) / wsum;
   }
   return (r & 255u) | ((g & 255u) << 8u) | ((b & 255u) << 16u) | ((a & 255u) << 24u);
}

uint32_t
gb_conv2d_ref(bool bilateral, const uint32_t *in, uint32_t *out, uint32_t w, uint32_t h)
{
   uint32_t *tmp = malloc((size_t)w * h * sizeof(uint32_t));
   for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
         tmp[y * w + x] = conv_pixel_1d(in, (int)x, (int)y, (int)w, (int)h, false, bilateral);
   for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++)
         out[y * w + x] = conv_pixel_1d(tmp, (int)x, (int)y, (int)w, (int)h, true, bilateral);
   free(tmp);
   return gb_crc32(out, (size_t)w * h * sizeof(uint32_t));
}

/* =========================================================================
 * compute.stereo CPU reference (450x375, pitch 512, 11x11 dense SAD)
 * ========================================================================= */

static inline uint8_t
stereo_byte(const uint32_t *packed, uint32_t x, uint32_t y)
{
   uint32_t off = y * GB_STEREO_PITCH + x;
   return (uint8_t)((packed[off >> 2u] >> ((off & 3u) * 8u)) & 255u);
}

void
gb_stereo_fill_inputs(uint32_t *left_packed, uint32_t *right_packed, uint32_t disp)
{
   const size_t nbytes = (size_t)GB_STEREO_PITCH * GB_STEREO_H;
   uint8_t *L = calloc(1, nbytes);
   uint8_t *R = calloc(1, nbytes);
   for (uint32_t y = 0; y < GB_STEREO_H; y++) {
      for (uint32_t x = 0; x < GB_STEREO_W; x++) {
         uint32_t h = (x * 374761393u + y * 668265263u) ^ 0x5bf03635u;
         h = (h ^ (h >> 13u)) * 1274126177u;
         L[y * GB_STEREO_PITCH + x] = (uint8_t)((h >> 24u) ^ ((x * 5u + y * 3u) & 255u));
      }
   }
   uint32_t dmax = disp > 4u ? disp - 4u : 1u;
   for (uint32_t y = 0; y < GB_STEREO_H; y++) {
      for (uint32_t x = 0; x < GB_STEREO_W; x++) {
         uint32_t dtrue = (((x / 32u) + (y / 32u) * 3u) * 7u + 3u) % dmax;
         uint32_t sx = x + dtrue < GB_STEREO_W ? x + dtrue : GB_STEREO_W - 1u;
         R[y * GB_STEREO_PITCH + x] = L[y * GB_STEREO_PITCH + sx];
      }
   }
   memcpy(left_packed, L, nbytes);
   memcpy(right_packed, R, nbytes);
   free(L);
   free(R);
}

uint32_t
gb_stereo_ref(const uint32_t *left_packed, const uint32_t *right_packed, uint32_t disp, uint32_t *out)
{
   const int W = (int)GB_STEREO_W, H = (int)GB_STEREO_H, RAD = (int)GB_STEREO_RADIUS;
   memset(out, 0, (size_t)GB_STEREO_PITCH * GB_STEREO_H * sizeof(uint32_t));

   /* Fast integral-image SAD oracle matching exact edge-clamped 11x11 window. */
   const int PW = W + 2 * RAD;
   const int PH = H + 2 * RAD;
   uint32_t *best_sad = malloc((size_t)W * H * sizeof(uint32_t));
   uint32_t *best_d = calloc((size_t)W * H, sizeof(uint32_t));
   uint32_t *sat = calloc((size_t)(PW + 1) * (PH + 1), sizeof(uint32_t));
   for (int i = 0; i < W * H; i++)
      best_sad[i] = 0x7fffffffu;

   for (uint32_t d = 0; d < disp; d++) {
      for (int py = 0; py < PH; py++) {
         int sy = py - RAD;
         if (sy < 0) sy = 0;
         if (sy >= H) sy = H - 1;
         uint32_t row = 0;
         for (int px = 0; px < PW; px++) {
            int sx = px - RAD;
            int lx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);
            int rx = sx - (int)d;
            if (rx < 0) rx = 0;
            if (rx >= W) rx = W - 1;
            int lv = stereo_byte(left_packed, (uint32_t)lx, (uint32_t)sy);
            int rv = stereo_byte(right_packed, (uint32_t)rx, (uint32_t)sy);
            row += (uint32_t)(lv > rv ? lv - rv : rv - lv);
            sat[(py + 1) * (PW + 1) + (px + 1)] = sat[py * (PW + 1) + (px + 1)] + row;
         }
      }
      for (int y = 0; y < H; y++) {
         int y0 = y, y1 = y + 2 * RAD + 1;
         for (int x = 0; x < W; x++) {
            int x0 = x, x1 = x + 2 * RAD + 1;
            uint32_t sad = sat[y1 * (PW + 1) + x1] - sat[y0 * (PW + 1) + x1] -
                           sat[y1 * (PW + 1) + x0] + sat[y0 * (PW + 1) + x0];
            int idx = y * W + x;
            if (sad < best_sad[idx]) {
               best_sad[idx] = sad;
               best_d[idx] = d;
            }
         }
      }
   }
   for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
         int idx = y * W + x;
         out[(uint32_t)y * GB_STEREO_PITCH + (uint32_t)x] =
            (best_d[idx] & 0xffu) | ((best_sad[idx] & 0xffffffu) << 8u);
      }
   free(best_sad);
   free(best_d);
   free(sat);
   return gb_crc32(out, (size_t)GB_STEREO_PITCH * GB_STEREO_H * sizeof(uint32_t));
}

/* =========================================================================
 * compute.edge CPU reference (Sobel+NMS and Harris+NMS)
 * ========================================================================= */

void
gb_edge_fill_input(uint32_t *img, uint32_t w, uint32_t h)
{
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         /* Crisp geometric checker + circles + diagonal bars for rich edges & corners. */
         uint32_t bx = (x >> 4u) & 1u, by = (y >> 4u) & 1u;
         uint32_t v = (bx ^ by) ? 220u : 30u;
         int dx = (int)(x % 64u) - 32, dy = (int)(y % 64u) - 32;
         if (dx * dx + dy * dy < 225)
            v = 255u - v;
         uint32_t hsh = (x * 73856093u ^ y * 19349663u) >> 28u;
         v = (v + hsh) & 255u;
         img[y * w + x] = v;
      }
   }
}

static inline int
clampi(int v, int lo, int hi)
{
   return v < lo ? lo : (v > hi ? hi : v);
}

uint32_t
gb_edge_ref(bool harris, const uint32_t *in, uint32_t *out, uint32_t *corner_count_out, uint32_t w, uint32_t h)
{
   const int W = (int)w, H = (int)h;
   const size_t npix = (size_t)w * h;
   uint32_t *grad = malloc(npix * sizeof(uint32_t));
   uint32_t *mag = malloc(npix * sizeof(uint32_t));

   /* Stage 1: Sobel */
   for (int y = 0; y < H; y++) {
      int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);
      for (int x = 0; x < W; x++) {
         int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
         int p00 = (int)in[ym * W + xm], p01 = (int)in[ym * W + x], p02 = (int)in[ym * W + xp];
         int p10 = (int)in[y * W + xm],  p12 = (int)in[y * W + xp];
         int p20 = (int)in[yp * W + xm], p21 = (int)in[yp * W + x], p22 = (int)in[yp * W + xp];
         int gx = -p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22;
         int gy = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
         grad[y * W + x] = ((uint32_t)(gx + 2048) & 0xfffu) | (((uint32_t)(gy + 2048) & 0xfffu) << 12u);
         mag[y * W + x] = (uint32_t)((gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy));
      }
   }

   if (!harris) {
      /* Stage 2: Directional NMS + dual threshold */
      for (int y = 0; y < H; y++) {
         int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);
         for (int x = 0; x < W; x++) {
            int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
            uint32_t g = grad[y * W + x];
            int gx = (int)(g & 0xfffu) - 2048;
            int gy = (int)((g >> 12u) & 0xfffu) - 2048;
            int ax = gx < 0 ? -gx : gx, ay = gy < 0 ? -gy : gy;
            uint32_t m = mag[y * W + x];
            uint32_t n1, n2;
            if (ax >= 2 * ay) {
               n1 = mag[y * W + xm]; n2 = mag[y * W + xp];
            } else if (ay >= 2 * ax) {
               n1 = mag[ym * W + x]; n2 = mag[yp * W + x];
            } else if ((gx ^ gy) >= 0) {
               n1 = mag[ym * W + xm]; n2 = mag[yp * W + xp];
            } else {
               n1 = mag[ym * W + xp]; n2 = mag[yp * W + xm];
            }
            uint32_t kept = (m >= n1 && m >= n2) ? m : 0u;
            out[y * W + x] = kept >= 180u ? 255u : (kept >= 90u ? 128u : 0u);
         }
      }
      if (corner_count_out) *corner_count_out = 0;
   } else {
      /* Stage 2: Harris structure tensor + response */
      int32_t *resp = malloc(npix * sizeof(int32_t));
      for (int y = 0; y < H; y++) {
         for (int x = 0; x < W; x++) {
            int sxx = 0, syy = 0, sxy = 0;
            for (int dy = -1; dy <= 1; dy++) {
               int yy = clampi(y + dy, 0, H - 1);
               for (int dx = -1; dx <= 1; dx++) {
                  int xx = clampi(x + dx, 0, W - 1);
                  uint32_t g = grad[yy * W + xx];
                  int gx = ((int)(g & 0xfffu) - 2048) >> 2;
                  int gy = ((int)((g >> 12u) & 0xfffu) - 2048) >> 2;
                  sxx += gx * gx;
                  syy += gy * gy;
                  sxy += gx * gy;
               }
            }
            sxx >>= 4; syy >>= 4; sxy >>= 4;
            int det = sxx * syy - sxy * sxy;
            int tr = sxx + syy;
            int r = det - ((tr * tr) >> 4);
            resp[y * W + x] = r > 0 ? r : 0;
         }
      }
      /* Stage 3: 3x3 local maximum suppression + corner count */
      uint32_t corners = 0;
      for (int y = 0; y < H; y++) {
         int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);
         for (int x = 0; x < W; x++) {
            int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
            int c = resp[y * W + x];
            bool peak = c > 4096 &&
               c > resp[ym * W + xm] && c > resp[ym * W + x] && c > resp[ym * W + xp] &&
               c > resp[y  * W + xm] && c >= resp[y  * W + xp] &&
               c >= resp[yp * W + xm] && c >= resp[yp * W + x] && c >= resp[yp * W + xp];
            if (peak) {
               out[y * W + x] = 255u;
               corners++;
            } else {
               out[y * W + x] = 0u;
            }
         }
      }
      if (corner_count_out) *corner_count_out = corners;
      free(resp);
   }

   free(grad);
   free(mag);
   return gb_crc32(out, npix * sizeof(uint32_t));
}

/* =========================================================================
 * render.mdi CPU reference (1024x1024 exact grid of 10x10 colored quads)
 * ========================================================================= */

uint32_t
gb_mdi_tex_color(uint32_t tex_id)
{
   uint32_t h = (tex_id + 1u) * 2654435761u;
   h ^= h >> 16u;
   h *= 0x7feb352du;
   uint32_t r = 32u + ((h >> 0u) & 223u);
   uint32_t g = 32u + ((h >> 8u) & 223u);
   uint32_t b = 32u + ((h >> 16u) & 223u);
   return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

void
gb_mdi_fill_tex(uint32_t tex_id, uint32_t *rgba16x16)
{
   uint32_t c = gb_mdi_tex_color(tex_id);
   for (uint32_t i = 0; i < GB_MDI_TEX_DIM * GB_MDI_TEX_DIM; i++)
      rgba16x16[i] = c;
}

uint32_t
gb_mdi_ref_fb(uint32_t draws, uint32_t bindless, uint32_t *fb_rgba8)
{
   const uint32_t clear_rgba = 0xff000000u; /* (0, 0, 0, 255) */
   for (uint32_t i = 0; i < GB_MDI_DIM * GB_MDI_DIM; i++)
      fb_rgba8[i] = clear_rgba;
   for (uint32_t d = 0; d < draws; d++) {
      uint32_t cell = d % 10000u;
      uint32_t cx = cell % 100u;
      uint32_t cy = cell / 100u;
      uint32_t col = gb_mdi_tex_color(d % bindless);
      uint32_t x0 = cx * 10u, y0 = cy * 10u;
      for (uint32_t y = y0; y < y0 + 10u; y++)
         for (uint32_t x = x0; x < x0 + 10u; x++)
            fb_rgba8[y * GB_MDI_DIM + x] = col;
   }
   return gb_crc32(fb_rgba8, (size_t)GB_MDI_DIM * GB_MDI_DIM * sizeof(uint32_t));
}

/* =========================================================================
 * render.deferred CPU lighting oracle (tolerance-verified + CRC-32)
 * ========================================================================= */

static inline float clampf01(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

static void
deferred_eval_pixel(float u, float v, uint8_t out_rgb[3])
{
   /* Front-most layer is layer 0 (z = 0.25) covering the entire viewport in a
    * 32x32 quad grid (quads 0..1023). */
   uint32_t qx = (uint32_t)(u * 32.0f);
   uint32_t qy = (uint32_t)(v * 32.0f);
   if (qx >= 32u) qx = 31u;
   if (qy >= 32u) qy = 31u;
   uint32_t qid = qy * 32u + qx;

   float ar = 0.2f + 0.7f * (float)((qid * 37u) & 255u) / 255.0f;
   float ag = 0.2f + 0.7f * (float)((qid * 73u + 51u) & 255u) / 255.0f;
   float ab = 0.2f + 0.7f * (float)((qid * 131u + 17u) & 255u) / 255.0f;
   float rough = 0.15f + 0.7f * (float)((qid * 19u) & 255u) / 255.0f;
   float nx = (float)((qid * 53u) & 255u) / 255.0f * 0.8f - 0.4f;
   float ny = (float)((qid * 97u) & 255u) / 255.0f * 0.8f - 0.4f;
   float nz = 1.0f;
   float nlen = sqrtf(nx * nx + ny * ny + nz * nz);
   nx /= nlen; ny /= nlen; nz /= nlen;
   float metal = (qid & 3u) == 0u ? 0.8f : 0.1f;
   float er = 0.05f * (float)((qid * 11u) & 255u) / 255.0f;
   float eg = 0.04f * (float)((qid * 23u) & 255u) / 255.0f;
   float eb = 0.06f * (float)((qid * 41u) & 255u) / 255.0f;
   float ao = 0.6f + 0.4f * (float)((qid * 29u) & 255u) / 255.0f;
   float linz = 0.25f;

   /* Match exact lighting math in deferred_tile.frag / deferred_dram.frag / wave1.metal */
   const float ldir[4][3] = {
      { 0.57735027f,  0.57735027f, 0.57735027f},
      {-0.53452248f,  0.26726124f, 0.80178373f},
      { 0.26726124f, -0.53452248f, 0.80178373f},
      {-0.40824829f, -0.40824829f, 0.81649658f},
   };
   const float lcol[4][3] = {
      {0.95f, 0.90f, 0.85f},
      {0.40f, 0.55f, 0.85f},
      {0.75f, 0.45f, 0.30f},
      {0.35f, 0.70f, 0.50f},
   };
   float cr = ar * 0.12f * ao + er;
   float cg = ag * 0.12f * ao + eg;
   float cb = ab * 0.12f * ao + eb;
   float shin = 8.0f + (1.0f - rough) * 24.0f;
   float atten = 1.0f / (1.0f + linz * linz);
   for (int i = 0; i < 4; i++) {
      float ndl = nx * ldir[i][0] + ny * ldir[i][1] + nz * ldir[i][2];
      if (ndl < 0.0f) ndl = 0.0f;
      float hx = ldir[i][0], hy = ldir[i][1], hz = ldir[i][2] + 1.0f;
      float hl = sqrtf(hx * hx + hy * hy + hz * hz);
      float ndh = (nx * hx + ny * hy + nz * hz) / hl;
      if (ndh < 0.0f) ndh = 0.0f;
      float spec = powf(ndh, shin) * (0.2f + 0.8f * metal);
      float diff = ndl * (1.0f - 0.5f * metal);
      cr += (ar * diff + spec) * lcol[i][0] * atten;
      cg += (ag * diff + spec) * lcol[i][1] * atten;
      cb += (ab * diff + spec) * lcol[i][2] * atten;
   }
   out_rgb[0] = (uint8_t)(clampf01(cr) * 255.0f + 0.5f);
   out_rgb[1] = (uint8_t)(clampf01(cg) * 255.0f + 0.5f);
   out_rgb[2] = (uint8_t)(clampf01(cb) * 255.0f + 0.5f);
}

void
gb_deferred_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, struct gb_check *chk)
{
   chk->done = true;
   chk->crc = gb_crc32(rgba8, (size_t)w * h * 4u);
   chk->expected = chk->crc; /* tolerance-validated analytical check */
   chk->valid = true;

   /* Sample the center of 64 interior cells (8x8 across the 32x32 grid) to avoid
    * quad boundary pixels under 4x MSAA. */
   for (uint32_t gy = 0; gy < 8u; gy++) {
      for (uint32_t gx = 0; gx < 8u; gx++) {
         uint32_t qx = gx * 4u + 1u;
         uint32_t qy = gy * 4u + 1u;
         float u = ((float)qx + 0.5f) / 32.0f;
         float v = ((float)qy + 0.5f) / 32.0f;
         uint32_t px = (uint32_t)(u * (float)w);
         uint32_t py = (uint32_t)(v * (float)h);
         if (px >= w) px = w - 1u;
         if (py >= h) py = h - 1u;
         uint8_t want[3];
         deferred_eval_pixel(u, v, want);
         const uint8_t *got = rgba8 + ((size_t)py * w + px) * 4u;
         for (int c = 0; c < 3; c++) {
            int diff = (int)got[c] - (int)want[c];
            if (diff < 0) diff = -diff;
            if (diff > 4) {
               gb_log("deferred mismatch at (%u,%u) ch %d: got %u want %u", px, py, c, got[c], want[c]);
               chk->valid = false;
               chk->expected = 0u;
               return;
            }
         }
      }
   }
}
