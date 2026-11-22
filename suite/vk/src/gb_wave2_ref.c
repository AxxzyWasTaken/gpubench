// SPDX-License-Identifier: GPL-2.0-only
#include "gb_wave2_ref.h"
#include "gb_params.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
hash_u32(uint32_t x)
{
   x = (x ^ 61u) ^ (x >> 16);
   x = x + (x << 3);
   x = x ^ (x >> 4);
   x = x * 0x27d4eb2du;
   x = x ^ (x >> 15);
   return x;
}

static uint16_t
f32_to_f16_exact_small(float v)
{
   /* Exact IEEE 754 binary16 encoding for small integers / dyadic values in
    * [-32, 32] (used for deterministic quantized GEMV inputs). */
   if (v == 0.0f)
      return 0u;
   uint32_t bits;
   memcpy(&bits, &v, 4);
   uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
   int exp = (int)((bits >> 23) & 0xffu) - 127 + 15;
   uint32_t mant = (bits >> 13) & 0x3ffu;
   if (exp <= 0)
      return sign;
   if (exp >= 31)
      return (uint16_t)(sign | 0x7c00u);
   return (uint16_t)(sign | ((uint16_t)exp << 10) | (uint16_t)mant);
}

static float
f16_to_f32_exact_small(uint16_t h)
{
   uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
   uint32_t exp = (h >> 10) & 0x1fu;
   uint32_t mant = h & 0x3ffu;
   if (exp == 0 && mant == 0) {
      float z = 0.0f;
      if (sign)
         z = -0.0f;
      return z;
   }
   uint32_t bits = sign | ((exp + 112u) << 23) | (mant << 13);
   float f;
   memcpy(&f, &bits, 4);
   return f;
}

/* ======================= 1. render.shadow ======================= */

void
gb_shadow_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, bool alpha_test, struct gb_check *chk)
{
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(rgba8, (size_t)w * h * 4u);
   chk->expected = chk->crc;

   int lit_bad = 0, shad_bad = 0, lit_n = 0, shad_n = 0;
   double alpha_shad_r_sum = 0.0;
   int alpha_shad_lit = 0, alpha_shad_dark = 0;

   /* Sample interior centers of cells across all 4 cascade quadrants. */
   for (uint32_t c = 0; c < 4u; c++) {
      uint32_t qx = c & 1u, qy = c >> 1;
      for (uint32_t gy = 2; gy < 30; gy += 3) {
         for (uint32_t gx = 2; gx < 30; gx += 3) {
            uint32_t px = qx * (w / 2u) + (uint32_t)(((gx + 0.5) / 32.0) * (w / 2u));
            uint32_t py = qy * (h / 2u) + (uint32_t)(((gy + 0.5) / 32.0) * (h / 2u));
            const uint8_t *p = rgba8 + ((size_t)py * w + px) * 4u;
            bool caster = (((gx + gy + c) & 1u) == 0u);
            if (!caster) {
               lit_n++;
               if (abs((int)p[0] - 230) > 6 || abs((int)p[1] - 191) > 6 || abs((int)p[2] - 191) > 6)
                  lit_bad++;
            } else if (!alpha_test) {
               shad_n++;
               if (abs((int)p[0] - 51) > 6 || abs((int)p[1] - 64) > 6 || abs((int)p[2] - 89) > 6)
                  shad_bad++;
            } else {
               for (int dy = -2; dy <= 2; dy++) {
                  for (int dx = -2; dx <= 2; dx++) {
                     const uint8_t *q = rgba8 + ((size_t)(py + dy) * w + (px + dx)) * 4u;
                     alpha_shad_r_sum += q[0];
                     if (q[0] > 180)
                        alpha_shad_lit++;
                     if (q[0] < 110)
                        alpha_shad_dark++;
                     shad_n++;
                  }
               }
            }
         }
      }
   }

   if (lit_bad > 0) {
      chk->valid = false;
      gb_log("render.shadow check failed: lit_cells_bad=%d/%d", lit_bad, lit_n);
      return;
   }
   if (!alpha_test) {
      if (shad_bad > 0) {
         chk->valid = false;
         gb_log("render.shadow check failed: shadow_cells_bad=%d/%d", shad_bad, shad_n);
         return;
      }
      chk->valid = true;
   } else {
      double mean_r = shad_n ? alpha_shad_r_sum / (double)shad_n : 0.0;
      chk->valid = (mean_r >= 85.0 && mean_r <= 195.0 && alpha_shad_lit > 0 && alpha_shad_dark > 0);
      if (!chk->valid)
         gb_log("render.shadow alpha_test check failed: mean_r=%.1f dark=%d lit=%d",
                mean_r, alpha_shad_dark, alpha_shad_lit);
   }
}

/* ======================= 2. compute.cull_mdi ======================= */

uint32_t
gb_cull_mdi_fill_objects(uint32_t objs, uint32_t visible_pct, struct gb_cull_obj *out)
{
   uint32_t vis_count = 0;
   for (uint32_t i = 0; i < objs; i++) {
      uint32_t h = hash_u32(i ^ 0x9e3779b9u);
      bool vis = (h % 100u) < visible_pct;
      if (vis) {
         out[i].cx = 0.5f * ((float)(i & 127u) / 128.0f);
         out[i].cy = 0.5f * ((float)((i >> 7) & 127u) / 128.0f);
         out[i].r = 0.125f;
         vis_count++;
      } else {
         out[i].cx = 3.0f;
         out[i].cy = 3.0f;
         out[i].r = 0.125f;
      }
      uint32_t r = 32u + (h & 191u);
      uint32_t g = 32u + ((h >> 8) & 191u);
      uint32_t b = 32u + ((h >> 16) & 191u);
      out[i].color_rgba8 = r | (g << 8) | (b << 16) | 0xff000000u;
   }
   return vis_count;
}

uint32_t
gb_cull_mdi_ref_fb(const struct gb_cull_obj *objs, uint32_t n_objs, uint32_t *fb_rgba8)
{
   const uint32_t dim = GB_CULL_DIM;
   for (uint32_t i = 0; i < dim * dim; i++)
      fb_rgba8[i] = 0xff000000u;
   for (uint32_t id = 0; id < n_objs; id++) {
      float cx = objs[id].cx, cy = objs[id].cy, r = objs[id].r;
      bool vis = (cx + r >= -1.0f) && (-cx + r >= -1.0f) && (cy + r >= -1.0f) && (-cy + r >= -1.0f) &&
                 (cx * cx + cy * cy < 2.0f);
      if (!vis)
         continue;
      uint32_t gx = id & 127u;
      uint32_t gy = (id >> 7) & 127u;
      uint32_t col = objs[id].color_rgba8;
      for (uint32_t y = gy * 8u; y < gy * 8u + 8u; y++)
         for (uint32_t x = gx * 8u; x < gx * 8u + 8u; x++)
            fb_rgba8[y * dim + x] = col;
   }
   return gb_crc32(fb_rgba8, (size_t)dim * dim * 4u);
}

/* ======================= 3. render.bindless_ubo ======================= */

static uint32_t
bubo_entry_rgba8(uint32_t bank_id, uint32_t idx)
{
   uint32_t h = hash_u32((bank_id << 16) | idx);
   uint32_t r = h & 255u;
   uint32_t g = (h >> 8) & 255u;
   uint32_t b = (h >> 16) & 255u;
   return r | (g << 8) | (b << 16) | 0xff000000u;
}

void
gb_bubo_fill_bank(uint32_t bank_id, float *rgba_1024)
{
   for (uint32_t i = 0; i < GB_BUBO_ENTRIES; i++) {
      uint32_t c = bubo_entry_rgba8(bank_id, i);
      rgba_1024[i * 4u + 0u] = (float)(c & 255u) / 255.0f;
      rgba_1024[i * 4u + 1u] = (float)((c >> 8) & 255u) / 255.0f;
      rgba_1024[i * 4u + 2u] = (float)((c >> 16) & 255u) / 255.0f;
      rgba_1024[i * 4u + 3u] = 1.0f;
   }
}

uint32_t
gb_bubo_ref_fb(uint32_t draws, uint32_t *fb_rgba8)
{
   const uint32_t dim = GB_BUBO_DIM;
   for (uint32_t i = 0; i < dim * dim; i++)
      fb_rgba8[i] = 0xff000000u;
   for (uint32_t d = 0; d < draws; d++) {
      uint32_t tx = d & 63u;
      uint32_t ty = (d >> 6) & 31u;
      uint32_t bank = d & (GB_BUBO_BANKS - 1u);
      for (uint32_t y = ty * 32u; y < ty * 32u + 32u; y++) {
         for (uint32_t x = tx * 16u; x < tx * 16u + 16u; x++) {
            uint32_t idx0 = (d * 13u + ((x & 15u) ^ (y & 15u))) & 1023u;
            uint32_t c0 = bubo_entry_rgba8(bank, idx0);
            uint32_t idx1 = (idx0 + (c0 & 255u)) & 1023u;
            uint32_t c1 = bubo_entry_rgba8(bank, idx1);
            uint32_t idx2 = (idx1 + ((c1 >> 8) & 255u)) & 1023u;
            uint32_t c2 = bubo_entry_rgba8(bank, idx2);
            uint32_t idx3 = (idx2 + ((c2 >> 16) & 255u)) & 1023u;
            fb_rgba8[y * dim + x] = bubo_entry_rgba8(bank, idx3);
         }
      }
   }
   return gb_crc32(fb_rgba8, (size_t)dim * dim * 4u);
}

/* ======================= 4. compute.fp16 ======================= */

void
gb_fp16_gemv_fill(uint32_t *w_packed, uint32_t *x_fp16_pairs)
{
   const uint32_t words = (GB_FP16_GEMV_M * GB_FP16_GEMV_K) / 4u;
   for (uint32_t i = 0; i < words; i++) {
      uint32_t h = hash_u32(i ^ 0x517cc1b7u);
      uint32_t b0 = h & 3u;
      uint32_t b1 = (h >> 4) & 3u;
      uint32_t b2 = (h >> 8) & 3u;
      uint32_t b3 = (h >> 12) & 3u;
      w_packed[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
   }
   static const float xvals[5] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
   for (uint32_t i = 0; i < GB_FP16_GEMV_K / 2u; i++) {
      uint32_t h = hash_u32(i ^ 0x243f6a88u);
      uint16_t lo = f32_to_f16_exact_small(xvals[h % 5u]);
      uint16_t hi = f32_to_f16_exact_small(xvals[(h >> 8) % 5u]);
      x_fp16_pairs[i] = (uint32_t)lo | ((uint32_t)hi << 16);
   }
}

uint32_t
gb_fp16_gemv_ref(const uint32_t *w_packed, const uint32_t *x_fp16_pairs, float *out_m)
{
   float x[GB_FP16_GEMV_K];
   for (uint32_t i = 0; i < GB_FP16_GEMV_K / 2u; i++) {
      x[2u * i + 0u] = f16_to_f32_exact_small((uint16_t)(x_fp16_pairs[i] & 0xffffu));
      x[2u * i + 1u] = f16_to_f32_exact_small((uint16_t)(x_fp16_pairs[i] >> 16));
   }
   const uint32_t row_words = GB_FP16_GEMV_K / 4u;
   for (uint32_t r = 0; r < GB_FP16_GEMV_M; r++) {
      float acc = 0.0f;
      const uint32_t *rp = w_packed + (size_t)r * row_words;
      for (uint32_t w = 0; w < row_words; w++) {
         uint32_t pk = rp[w];
         float w0 = (float)(pk & 0xffu);
         float w1 = (float)((pk >> 8) & 0xffu);
         float w2 = (float)((pk >> 16) & 0xffu);
         float w3 = (float)((pk >> 24) & 0xffu);
         uint32_t k = w * 4u;
         acc += w0 * x[k + 0u] + w1 * x[k + 1u] + w2 * x[k + 2u] + w3 * x[k + 3u];
      }
      out_m[r] = acc;
   }
   return gb_crc32(out_m, (size_t)GB_FP16_GEMV_M * sizeof(float));
}

uint32_t
gb_fp16_alumix_ref(uint32_t nthreads, uint32_t iters, float a, float b, float *out)
{
   for (uint32_t id = 0; id < nthreads; id++) {
      float h[4][4], f[2][4];
      for (uint32_t k = 0; k < 4u; k++) {
         float base0 = (float)((id + k) & 7u) * b;
         float base1 = (float)((id + k + 2u) & 7u) * b;
         h[0][k] = base0;
         h[1][k] = base1;
         h[2][k] = -base0;
         h[3][k] = -base1;
         f[0][k] = base0 + b;
         f[1][k] = base1 - b;
      }
      float sgn = (id & 1u) ? b : -b;
      float c[4][4], d[4][4];
      for (uint32_t j = 0; j < 4u; j++) {
         for (uint32_t k = 0; k < 4u; k++) {
            c[j][k] = (float)((id + j * 3u + k) & 7u) * b;
            d[j][k] = c[j][k] + ((j & 1u) ? -sgn : sgn);
         }
      }
      for (uint32_t i = 0; i < iters; i++) {
         for (uint32_t j = 0; j < 4u; j++) {
            for (uint32_t k = 0; k < 4u; k++) {
               h[j][k] = fmaf(h[j][k], a, c[j][k]);
               h[j][k] = fmaf(h[j][k], a, d[j][k]);
            }
         }
         for (uint32_t j = 0; j < 2u; j++) {
            for (uint32_t k = 0; k < 4u; k++) {
               f[j][k] = fmaf(f[j][k], a, c[j][k]);
               f[j][k] = fmaf(f[j][k], a, d[j][k]);
            }
         }
      }
      float sum = 0.0f;
      for (uint32_t k = 0; k < 4u; k++)
         sum += h[0][k] + h[1][k] + h[2][k] + h[3][k] + f[0][k] + f[1][k];
      out[id] = sum;
   }
   return gb_crc32(out, (size_t)nthreads * sizeof(float));
}

/* ======================= 5. transfer.ops ======================= */

uint32_t
gb_transfer_clear_color_u32(void)
{
   /* RGBA = (192, 128, 64, 255) = 0xff4080c0u in little-endian RGBA8 */
   return 0xff4080c0u;
}

uint32_t
gb_transfer_clear_ref_crc(uint32_t w, uint32_t h)
{
   size_t n = (size_t)w * h;
   uint32_t *buf = malloc(n * 4u);
   uint32_t col = gb_transfer_clear_color_u32();
   for (size_t i = 0; i < n; i++)
      buf[i] = col;
   uint32_t crc = gb_crc32(buf, n * 4u);
   free(buf);
   return crc;
}

void
gb_transfer_fill_rgba(uint32_t w, uint32_t h, uint32_t *rgba)
{
   for (uint32_t y = 0; y < h; y++) {
      for (uint32_t x = 0; x < w; x++) {
         uint32_t r = (x * 3u + y) & 255u;
         uint32_t g = (y * 5u + (x >> 1)) & 255u;
         uint32_t b = ((x ^ y) * 7u) & 255u;
         rgba[(size_t)y * w + x] = r | (g << 8) | (b << 16) | 0xff000000u;
      }
   }
}

void
gb_transfer_fill_mip0(uint32_t w, uint32_t h, uint32_t *rgba)
{
   for (uint32_t y = 0; y < h; y++) {
      uint32_t g = (y * 255u) / (h - 1u);
      for (uint32_t x = 0; x < w; x++) {
         uint32_t r = (x * 255u) / (w - 1u);
         rgba[(size_t)y * w + x] = r | (g << 8) | (128u << 16) | 0xff000000u;
      }
   }
}

void
gb_transfer_check_mipchain(const uint8_t *mip4_rgba, uint32_t m4w, uint32_t m4h,
                           const uint8_t *mip_last_rgba, struct gb_check *chk)
{
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(mip4_rgba, (size_t)m4w * m4h * 4u);
   chk->expected = chk->crc;

   int last_ok = abs((int)mip_last_rgba[0] - 128) <= 8 && abs((int)mip_last_rgba[1] - 128) <= 8 &&
                 abs((int)mip_last_rgba[2] - 128) <= 4 && mip_last_rgba[3] >= 250;
   const uint8_t *tl = mip4_rgba;
   const uint8_t *tr = mip4_rgba + (size_t)(m4w - 1u) * 4u;
   const uint8_t *bl = mip4_rgba + (size_t)(m4h - 1u) * m4w * 4u;
   const uint8_t *br = mip4_rgba + ((size_t)(m4h - 1u) * m4w + (m4w - 1u)) * 4u;
   int ramp_ok = (tl[0] < 24 && tl[1] < 24) && (tr[0] > 230 && tr[1] < 24) &&
                 (bl[0] < 24 && bl[1] > 230) && (br[0] > 230 && br[1] > 230) &&
                 abs((int)tl[2] - 128) <= 4 && abs((int)br[2] - 128) <= 4;
   chk->valid = last_ok && ramp_ok;
   if (!chk->valid)
      gb_log("transfer.ops mipchain check failed: mip_last=(%u,%u,%u,%u) m4_tl=(%u,%u) m4_br=(%u,%u)",
             mip_last_rgba[0], mip_last_rgba[1], mip_last_rgba[2], mip_last_rgba[3],
             tl[0], tl[1], br[0], br[1]);
}

/* ======================= 6. render.postfx ======================= */

void
gb_postfx_check_fb(const uint8_t *rgba8, uint32_t w, uint32_t h, struct gb_check *chk)
{
   memset(chk, 0, sizeof(*chk));
   chk->done = true;
   chk->crc = gb_crc32(rgba8, (size_t)w * h * 4u);
   chk->expected = chk->crc;

   const uint8_t *ctr = rgba8 + ((size_t)(h / 2u) * w + (w / 2u)) * 4u;
   const uint8_t *mid = rgba8 + ((size_t)(h / 4u) * w + (w / 4u)) * 4u;
   const uint8_t *cor = rgba8 + ((size_t)(h / 20u) * w + (w / 20u)) * 4u;
   double ctr_l = 0.2126 * ctr[0] + 0.7152 * ctr[1] + 0.0722 * ctr[2];
   double mid_l = 0.2126 * mid[0] + 0.7152 * mid[1] + 0.0722 * mid[2];
   double cor_l = 0.2126 * cor[0] + 0.7152 * cor[1] + 0.0722 * cor[2];

   chk->valid = (ctr[0] >= 200 && ctr[1] >= 170) && (ctr_l > mid_l + 30.0) &&
                (mid_l > cor_l + 5.0) && (cor_l >= 15.0 && cor_l <= 210.0) && ctr[3] == 255;
   if (!chk->valid)
      gb_log("render.postfx check failed: ctr=(%u,%u,%u,%u) mid_l=%.1f cor_l=%.1f",
             ctr[0], ctr[1], ctr[2], ctr[3], mid_l, cor_l);
}
